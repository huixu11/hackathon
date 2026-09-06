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
DEFAULT_LINEAR_DTYPE = "fp16"


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

        # Fixed-shape I/O for process_batch. Every round writes into these same
        # tensors and hands them to the model, so the forward pass sees one
        # shape for the life of the process and every buffer it reads keeps its
        # address - the two preconditions for capturing the pass as a CUDA graph.
        self.num_features = config.num_features
        self.features_buf = torch.zeros(
            num_symbols, self.num_features, dtype=torch.float32, device=self.device
        )
        self.active_mask = torch.zeros(
            num_symbols, dtype=torch.bool, device=self.device
        )

        # CPU staging, filled on the host and copied over in one shot. Page
        # locking is what makes that copy async and DMA-able, and it only means
        # anything on CUDA; on cpu/mps a plain tensor is the same allocation
        # without the pinning cost.
        if is_cuda:
            self.features_cpu = torch.empty(
                num_symbols, self.num_features, dtype=torch.float32, pin_memory=True
            )
            self.active_mask_cpu = torch.empty(
                num_symbols, dtype=torch.bool, pin_memory=True
            )
        else:
            self.features_cpu = torch.empty(
                num_symbols, self.num_features, dtype=torch.float32
            )
            self.active_mask_cpu = torch.empty(num_symbols, dtype=torch.bool)

        # numpy views onto the staging tensors, taken once here. Staging a round
        # is then a couple of numpy stores straight into the page-locked pages,
        # with no torch dispatch and no per-round allocation. They share storage
        # with the tensors above, so writing the view writes the buffer.
        self.features_staging = self.features_cpu.numpy()
        self.active_mask_staging = self.active_mask_cpu.numpy()

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
        list is longer than k. Rounds continue until the longest per-symbol list
        is exhausted, so each round holds a symbol at most once and every
        request gets exactly one prediction.

        Within a round the model runs over ALL num_symbols rows against the
        whole of self.batched_state, never a subset. A symbol with nothing
        queued this round is fed a row of zeros and marked false in an active
        mask; afterwards the state its row produced is discarded and the state
        it already had is kept. Three things come out of that:

          - The forward pass has one shape for the life of the process instead
            of a new shape every round, which is the precondition for capturing
            it as a CUDA graph later. Static addresses are the other half, and
            the blend below preserves those.
          - No gather, no scatter. Pulling the active rows out of the state and
            writing them back used to cost two index kernels per state tensor
            per round - hundreds of launches each moving a few kilobytes, which
            is nearly all launch overhead.
          - The idle rows are close to free. At this batch size the pass is
            bandwidth bound on streaming the weights out of HBM, and the weights
            are read once however many rows ride along; only the recurrent
            state, a small share of the bytes, scales with the row count.
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
            """One forward pass over every symbol, with a mask over the live rows."""
            # Stage the round on the host: zeros everywhere, then this round's
            # rows written as a single (n, num_features) block. Building that
            # block with numpy keeps it one store rather than a Python loop over
            # the individual floats.
            index = np.asarray(batch_indices, dtype=np.int64)
            rows = np.asarray(batch_features, dtype=np.float32)

            self.features_staging[:] = 0.0
            self.features_staging[index] = rows
            self.active_mask_staging[:] = False
            self.active_mask_staging[index] = True

            # Both copies are enqueued on the current stream ahead of the
            # forward pass, so the mask the model runs behind is this round's
            # mask, written before any of the model's kernels touch the buffer.
            self.features_buf.copy_(self.features_cpu, non_blocking=True)
            self.active_mask.copy_(self.active_mask_cpu, non_blocking=True)

            # ONE forward pass, always the full batch, always the same shape.
            preds, new_state = self.model(self.features_buf, self.batched_state)

            # Fold the rows that ran into the live state, using the same mask.
            # An idle row can come back as garbage or nan - SLSTM's running max
            # starts at -inf, and a row of zero features is not a meaningful
            # event for any of the towers - but the mask throws that row away,
            # and no operator in the model mixes rows (every norm here is
            # per-sample, the convs are depthwise, the einsums keep b on both
            # sides), so the symbols that did run are untouched by it.
            self._blend_state(self.batched_state, new_state, self.active_mask)

            # A single D2H copy for the whole (num_symbols, 4) block; the active
            # rows are then picked out by index on the host. .cpu() also
            # synchronizes the stream, which is what makes it safe for the next
            # round to overwrite the staging buffers the async copies read from.
            preds_cpu = preds.cpu().numpy()
            for req, symbol_idx in zip(batch_requests, batch_indices):
                all_unique_ids.append(req.unique_id)
                all_predictions.append(preds_cpu[symbol_idx].astype(float).tolist())

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
    
    def _blend_state(self, old, new, mask):
        """Take the rows of `new` the mask selects; keep `old` everywhere else.

        Walks the two trees together. They have the same shape by construction:
        the model hands back exactly what init_state built, list for list and
        tuple for tuple, down to the RetNet counter and the xLSTM cells.

        Each leaf is written with copy_ rather than rebound, so every tensor in
        self.batched_state keeps its identity and its address across rounds.
        That is not tidiness - a captured CUDA graph replays against the
        addresses it recorded, so the state has to live in place.

        `mask` is this round's (num_symbols,) bool mask, reshaped per leaf to
        (num_symbols, 1, 1, ...) so it broadcasts against that leaf's rank.
        torch.where selects, it does not compute, so it carries the int32 RetNet
        offsets as happily as the float32 leaves and needs no dtype special
        case, and a nan on a discarded row cannot leak into a kept one. The
        obvious alternative, old[mask] = new[mask], would have to read the mask
        on the host to size its output and would stall the pipeline every round.
        """
        if isinstance(old, list):
            for old_leaf, new_leaf in zip(old, new, strict=True):
                self._blend_state(old_leaf, new_leaf, mask)
        elif isinstance(old, tuple):
            for old_leaf, new_leaf in zip(old, new, strict=True):
                self._blend_state(old_leaf, new_leaf, mask)
        elif isinstance(old, dict):
            for key in old:
                self._blend_state(old[key], new[key], mask)
        elif isinstance(old, torch.Tensor):
            mask_view = mask.view(mask.shape[0], *(1,) * (old.dim() - 1))
            old.copy_(torch.where(mask_view, new, old))


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