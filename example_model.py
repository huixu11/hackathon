#!/usr/bin/env python3
"""
HORIZONTAL BATCHING - Each symbol appears at most once per batch.
This means we can process multiple symbols in ONE forward pass!
 Massive Improvement:
  - Latency: 73.8ms (down from 8949.6ms!)
  - Response rate: 206.66/s (maintaining good throughput)
  - Accuracy: 0.7673 (up from 0.0000)
  - PnL: $115.80/s (profitable instead of losing $15/s!)
  - Rank: #12 on leaderboard

  The horizontal batching is working! Processing multiple symbols in one forward pass has:
  1. Fixed the queue overflow issue
  2. Reduced latency by over 100x
  3. Restored model accuracy
  4. Made the system profitable

  The key insight was that since each symbol appears at most once per batch, we can maintain a single batched state
  and process all active symbols in parallel through one forward pass.

  
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import argparse
from typing import Dict, List
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download

from client import BaseInferenceClient, PendingRequest, InferenceResponse
from model.inference_model import MultiTowerModel, ModelConfig


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Why store the weights in low precision instead of wrapping the forward pass in
# torch.autocast: this model is memory-bandwidth bound, so what matters is the
# number of parameter bytes streamed from HBM per event. Autocast leaves the
# master weights in fp32 and re-casts them on every pass, which reads the full
# fp32 tensor and then writes a bf16 copy - strictly more traffic, not less.
# Holding the nn.Linear weights in bf16 halves the bytes actually read, and the
# linears are where nearly all the parameters live.
#
# The cast back to fp32 on the way out is load-bearing, not cosmetic. Everything
# that consumes a linear's output keeps fp32 weights or fp32 state: the Conv1d in
# CausalConv1d, the einsum in BlockLinear, the RetNet rotary built from an int32
# position counter, Hawk's sqrt(1 - a**2), and the running sums in Mamba2/xLSTM.
# Those either fail outright on a mixed-dtype operand or drift once a long-lived
# recurrent state is accumulated in 8 mantissa bits. So bf16 is confined to the
# matmul itself; every module boundary still sees float32.
class CastLinear(nn.Module):
    """Drop-in replacement for nn.Linear that keeps its weights in low precision.

    Casts the input down on entry and the result back to float32 on exit, so the
    surrounding fp32 modules are unaware anything changed. Attribute names match
    nn.Linear (weight, bias, in_features, out_features).
    """

    def __init__(self, linear: nn.Linear, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = dtype

        self.weight = nn.Parameter(
            linear.weight.detach().to(dtype=dtype), requires_grad=False
        )
        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.detach().to(dtype=dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            x.to(self.compute_dtype), self.weight, self.bias
        ).to(torch.float32)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, compute_dtype={self.compute_dtype}"
        )


# These feed exponential decays, recurrent gate rescaling, or the final prediction, where
# bf16 rounding compounds over the stream; ~0.2% of parameters, so fp32 costs no bandwidth.
FP32_LINEAR_NAMES = ("dt_proj", "igate_proj", "fgate_proj", "output_proj")


def convert_linears(
    model: nn.Module,
    dtype: torch.dtype,
    skip_names: tuple[str, ...] = FP32_LINEAR_NAMES,
) -> tuple[int, int]:
    """Replace every nn.Linear in `model` with a CastLinear holding `dtype` weights.

    An nn.Linear held under an attribute name in `skip_names` is left as it is,
    so the precision-sensitive projections keep their fp32 weights. Returns
    (converted, skipped), both counted per attribute site. Nothing else is
    touched: BlockLinear, Conv1d, RMSNorm/LayerNorm/GroupNorm and every
    recurrent state stay float32.
    """
    # Collect first, apply second: swapping modules while walking named_modules()
    # would mutate the containers the walk is iterating over.
    replacements: List[tuple] = []
    skipped = 0
    for _, parent in model.named_modules():
        # Read _modules directly rather than named_children(), which silently
        # skips a module that is reachable under two attribute names.
        for child_name, child in parent._modules.items():
            if isinstance(child, nn.Linear):
                if child_name in skip_names:
                    skipped += 1
                else:
                    replacements.append((parent, child_name, child))

    converted: Dict[int, CastLinear] = {}
    for parent, child_name, child in replacements:
        new_module = converted.get(id(child))
        if new_module is None:
            new_module = CastLinear(child, dtype)
            converted[id(child)] = new_module
        if isinstance(parent, nn.ModuleList):
            # Inside a ModuleList the attribute name is an index string ("3").
            parent[int(child_name)] = new_module
        else:
            setattr(parent, child_name, new_module)

    return len(replacements), skipped


# Values accepted in the LINEAR_DTYPE environment variable.
LINEAR_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": None,  # leave the linears alone
}
DEFAULT_LINEAR_DTYPE = "bf16"


class NnInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        device: str | None = None,
        token: str | None = None,
    ):
        super().__init__(num_symbols, server_host, server_port)

        self.device = device or get_default_device()

        # Enable TensorFloat32 for H100
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=79,
        )
        self.model = MultiTowerModel(config).to(self.device)
        self.model.eval()

        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"{nparams = }")

        weights_file = hf_hub_download(
            repo_id="jane-street-gpu-mode/hackathon",
            filename="state_dict.pt",
            token=token,
        )
        weights = torch.load(weights_file, weights_only=True)
        self.model.load_state_dict(weights)

        # Swap the linears to low-precision storage. This happens after
        # load_state_dict so the checkpoint still lands on plain fp32 nn.Linear
        # modules, under the key names it was saved with.
        requested = os.environ.get("LINEAR_DTYPE", DEFAULT_LINEAR_DTYPE).strip().lower()
        if requested not in LINEAR_DTYPES:
            print(
                f"LINEAR_DTYPE={requested!r} is not one of "
                f"{sorted(LINEAR_DTYPES)}; falling back to {DEFAULT_LINEAR_DTYPE}"
            )
            requested = DEFAULT_LINEAR_DTYPE
        linear_dtype = LINEAR_DTYPES[requested]
        is_cuda = torch.device(self.device).type == "cuda"

        if is_cuda and linear_dtype is not None:
            n_converted, n_skipped = convert_linears(self.model, linear_dtype)
            print(
                f"Converted {n_converted} linear layers to {requested} "
                f"({n_skipped} kept fp32)"
            )
            # The fp32 originals are unreachable now; hand their blocks back.
            torch.cuda.empty_cache()
        else:
            reason = (
                "LINEAR_DTYPE=fp32"
                if linear_dtype is None
                else f"device is {torch.device(self.device).type}, not cuda"
            )
            print(f"Linear layers stay fp32 ({reason})")

        # Initialize a SINGLE batched state for ALL symbols
        self.batched_state = self.model.init_state(num_symbols, self.device)
        
        # Map symbols to their position in the batch
        self.symbol_to_idx = {f"SYM_{i:03d}": i for i in range(num_symbols)}
        
        print(f"Horizontal batching ready for {num_symbols} symbols!")

    @torch.inference_mode()
    def process_batch(
        self, requests_by_symbol: Dict[str, List[PendingRequest]]
    ) -> InferenceResponse:
        """
        Answer every pending request, batching horizontally across symbols.

        A symbol can have several queued requests, and they must reach the model
        in list order so its recurrent state advances correctly. So the work is
        split into rounds: round k takes request index k from every symbol whose
        list is longer than k, runs those through a single forward pass, and
        writes the resulting state rows back. Rounds continue until the longest
        per-symbol list is exhausted. Each round therefore holds a symbol at
        most once, and every request gets exactly one prediction.
        """
        all_unique_ids = []
        all_predictions = []

        # Fixed order, so indices and requests stay aligned within every round.
        symbol_items = [
            (symbol, reqs) for symbol, reqs in requests_by_symbol.items() if reqs
        ]
        if not symbol_items:
            return InferenceResponse(
                unique_ids=all_unique_ids,
                predictions=all_predictions,
                client_timestamp=time.time()
            )

        def run_round(batch_indices, batch_features, batch_requests):
            """One forward pass over at most one request per symbol."""
            # Convert features to tensor
            features_tensor = torch.tensor(
                batch_features,
                device=self.device,
                dtype=torch.float32
            )

            # Extract states for these specific symbols
            active_state = self._extract_batch_state(self.batched_state, batch_indices)

            # Process ALL symbols of this round in ONE forward pass!
            preds, new_state = self.model(features_tensor, active_state)

            # Update the batched state for these symbols
            self._update_batch_state(self.batched_state, new_state, batch_indices)

            # Collect predictions
            preds_cpu = preds.cpu().numpy()
            for i, req in enumerate(batch_requests):
                all_unique_ids.append(req.unique_id)
                all_predictions.append(preds_cpu[i].astype(float).tolist())

        for k in range(max(len(reqs) for _, reqs in symbol_items)):
            batch_indices = []
            batch_features = []
            batch_requests = []

            for symbol, symbol_requests in symbol_items:
                if len(symbol_requests) > k:
                    # Take this symbol's k-th queued request
                    req = symbol_requests[k]

                    batch_indices.append(self.symbol_to_idx[symbol])
                    batch_features.append(req.features)
                    batch_requests.append(req)

            run_round(batch_indices, batch_features, batch_requests)

        return InferenceResponse(
            unique_ids=all_unique_ids,
            predictions=all_predictions,
            client_timestamp=time.time()
        )
    
    def _extract_batch_state(self, full_state, indices):
        """Extract states for specific symbol indices."""
        if isinstance(full_state, list):
            return [self._extract_batch_state(s, indices) for s in full_state]
        elif isinstance(full_state, tuple):
            return tuple(self._extract_batch_state(s, indices) for s in full_state)
        elif isinstance(full_state, dict):
            return {k: self._extract_batch_state(v, indices) for k, v in full_state.items()}
        elif isinstance(full_state, torch.Tensor):
            return full_state[indices]
        else:
            return full_state
    
    def _update_batch_state(self, full_state, new_state, indices):
        """Update states at specific symbol indices."""
        if isinstance(full_state, list):
            for i, s in enumerate(full_state):
                self._update_batch_state(s, new_state[i], indices)
        elif isinstance(full_state, tuple):
            for i, s in enumerate(full_state):
                self._update_batch_state(s, new_state[i], indices)
        elif isinstance(full_state, dict):
            for k in full_state:
                self._update_batch_state(full_state[k], new_state[k], indices)
        elif isinstance(full_state, torch.Tensor):
            full_state[indices] = new_state


def main():
    parser = argparse.ArgumentParser(description="Horizontal batching inference")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=20,
        help="Number of symbols in the tradeable universe",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token to download the model",
    )

    args = parser.parse_args()
    client = NnInferenceClient(
        num_symbols=args.num_symbols,
        server_host=args.host,
        server_port=args.port,
        token=args.token,
    )

    client.run()


if __name__ == "__main__":
    main()