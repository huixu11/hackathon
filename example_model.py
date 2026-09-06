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


def iter_state_tensors(state):
    """Yield every tensor leaf of a state tree, in _blend_state's walk order.

    Same container zoo as the blend sees: the towers are lists, a block's state
    is whatever its layer returned (a tuple for RetNet and xLSTM, a list for
    Mamba2 and Hawk, sometimes a bare tensor), and dicts are handled for the
    same reason _blend_state handles them - nothing in the model builds one
    today, but if one appears the two walks should still agree.
    """
    if isinstance(state, (list, tuple)):
        for leaf in state:
            yield from iter_state_tensors(leaf)
    elif isinstance(state, dict):
        for leaf in state.values():
            yield from iter_state_tensors(leaf)
    elif isinstance(state, torch.Tensor):
        yield state


# Values accepted in the STEP_MODE environment variable. All four produce a
# self._step_fn() that runs the forward pass and folds the new state in; they
# differ only in how much per-round launch overhead is paid to do it.
#
#   inductor   torch.compile(mode="reduce-overhead"): inductor's kernels,
#              replayed by cudagraph trees.
#   cudagraph  the eager kernels, captured once by hand into a CUDA graph.
#   compile    torch.compile with the default mode: inductor's kernels, but no
#              CUDA graphs, so each one is launched on its own. Slower than
#              inductor and meant for profiling - the profiler sees the fused
#              kernels individually, by name, instead of one graph launch.
#   eager      no capture at all - the reference path, and the only one that
#              runs anywhere but CUDA.
STEP_MODES = ("inductor", "cudagraph", "compile", "eager")
DEFAULT_STEP_MODE = "inductor"

# Enough passes to settle whatever is autotuned before the graph is recorded:
# inductor's kernel selection, cudnn.benchmark's algorithm search, and the
# allocator blocks the capture will bake in.
WARMUP_STEPS = 3


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

        # Last, because it captures the step: every buffer the step reads or
        # writes has to exist, and hold the address it will hold forever,
        # before the pass can be compiled or recorded.
        self._setup_step_fn(is_cuda)

    def _step(self) -> torch.Tensor:
        """One forward pass over the whole batch, plus the state blend.

        Takes no arguments and reads self.features_buf, self.batched_state and
        self.active_mask directly, so the addresses it runs against are the
        same on every call - that, plus the single fixed shape, is what lets
        the whole thing be captured once and replayed.

        Returns preds as the model produced it, deliberately not cloned: the
        caller copies it to the host immediately, and a clone per round would
        put an allocation back into the path this exists to shorten.
        """
        preds, new_state = self.model(
            self.features_buf, self.batched_state, mask=self.active_mask
        )

        # Fold the rows that ran into the live state, using the same mask the
        # model just ran under. The three big leaves - the xLSTM mLSTM cell,
        # the Mamba2 ssm_state, the RetNet recurrent_state - are already folded
        # by then: their cells took the mask and wrote them in place, and hand
        # back the very tensor that went in, so the blend below sees `new is
        # old` and skips them. It only writes the small leaves.
        # An idle row can come back as garbage or nan - SLSTM's running max
        # starts at -inf, and a row of zero features is not a meaningful event
        # for any of the towers - but the mask throws that row away, and no
        # operator in the model mixes rows (every norm here is per-sample, the
        # convs are depthwise, the einsums keep b on both sides), so the
        # symbols that did run are untouched by it.
        self._blend_state(self.batched_state, new_state, self.active_mask)
        return preds

    def _stage_idle_round(self) -> None:
        """Push a round in which no symbol is active into the device buffers.

        A mask of all false makes every masked write - the cells' in-place ones
        and _blend_state's - write old into old, so a step run this way leaves
        batched_state exactly as it found it, which is what makes it safe to
        run the warmup passes before the first real request has arrived.
        """
        self.features_staging[:] = 0.0
        self.active_mask_staging[:] = False
        self.features_buf.copy_(self.features_cpu, non_blocking=True)
        self.active_mask.copy_(self.active_mask_cpu, non_blocking=True)

    def _replay_graph(self) -> torch.Tensor:
        """Replay the captured step and hand back the tensor it writes into.

        Always the same tensor object at the same address, because that is what
        the graph recorded; the caller has to read it before the next replay.
        """
        self._graph.replay()
        return self._static_preds

    def _setup_step_fn(self, is_cuda: bool) -> None:
        """Choose how _step runs, and pay whatever warmup that choice costs."""
        mode = os.environ.get("STEP_MODE", DEFAULT_STEP_MODE).strip().lower()
        if mode not in STEP_MODES:
            print(
                f"STEP_MODE={mode!r} is not one of {sorted(STEP_MODES)}; "
                f"falling back to {DEFAULT_STEP_MODE}"
            )
            mode = DEFAULT_STEP_MODE

        if not is_cuda and mode != "eager":
            # Every mode but eager is CUDA-only here: mark_static_address
            # exists to feed cudagraph trees, CUDAGraph has no CPU or MPS
            # analogue, and the warmup loops synchronize the CUDA stream.
            print(
                f"STEP_MODE={mode} needs CUDA; device is "
                f"{torch.device(self.device).type}, so the step runs eager"
            )
            mode = "eager"

        started = time.perf_counter()

        if mode == "eager":
            self._step_fn = self._step

        elif mode in ("inductor", "compile"):
            # cudagraph trees will not replay against an input it thinks might
            # be reallocated, and it decides that from the address. These
            # buffers are written in place forever and never rebound, so say
            # so; without it the in-place mutation of the state makes it skip
            # CUDA graphs silently and only the kernel fusion is left. Under
            # "compile" there are no graphs to feed and the marks cost nothing.
            #
            # torch._dynamo is a lazily imported submodule, so pull it in
            # before reaching through torch for it. `from torch import` and not
            # `import torch._dynamo`, which would rebind `torch` as a local for
            # the whole of this method.
            from torch import _dynamo  # noqa: F401

            for tensor in iter_state_tensors(self.batched_state):
                torch._dynamo.mark_static_address(tensor)
            torch._dynamo.mark_static_address(self.features_buf)
            torch._dynamo.mark_static_address(self.active_mask)

            # Same compiler, same fused kernels; "inductor" adds cudagraph
            # trees on top and "compile" leaves them off, which is the whole
            # point of the latter - one launch per kernel is slower to run and
            # legible to the profiler, which sees each fused kernel by name
            # rather than a single graph replay.
            compile_kwargs = {"mode": "reduce-overhead"} if mode == "inductor" else {}
            self._step_fn = torch.compile(
                self._step, fullgraph=False, **compile_kwargs
            )

            # Warm up the way process_batch will call it. Dynamo guards on the
            # grad mode, so warming up outside inference_mode would throw the
            # compile away and pay it again on the first real request.
            with torch.inference_mode():
                for _ in range(WARMUP_STEPS):
                    # Mask all false, so every masked write copies old into old
                    # and batched_state comes out of the warmup unchanged.
                    self._stage_idle_round()
                    self._step_fn()
                    torch.cuda.synchronize()

        elif mode == "cudagraph":
            with torch.inference_mode():
                # Mask all false, so the blend copies old into old and
                # batched_state comes out of warmup and capture unchanged.
                self._stage_idle_round()

                # The warmup has to run on a side stream: capture records the
                # allocations the step makes, and the caching allocator only
                # hands out capture-safe blocks for a stream it has already
                # seen the step run on.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(WARMUP_STEPS):
                        self._step()
                torch.cuda.current_stream().wait_stream(s)

                # Capture under inference_mode too, so the kernels recorded are
                # the ones the real calls would have run.
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_preds = self._step()

            self._step_fn = self._replay_graph

        print(f"Step mode: {mode} (warmup {time.perf_counter() - started:.1f} s)")

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
            of a new shape every round, which is half of what capturing it as a
            CUDA graph needs. Static addresses are the other half, and the
            blend inside _step preserves those; _setup_step_fn does the
            capturing.
          - No gather, no scatter. Pulling the active rows out of the state and
            writing them back used to cost two index kernels per state tensor
            per round - hundreds of launches each moving a few kilobytes, which
            is nearly all launch overhead. The mask now goes into the model
            itself, and the three big leaves - one per block in three of the
            four towers: the xLSTM mLSTM cell, the Mamba2 ssm_state and the
            RetNet recurrent_state, together about 98% of the 5.6 GB of state
            at 39 rows - are written in place by their own cells, inside the
            kernel that computes them. _blend_state sees those come back as the
            same tensor object and skips them; it only handles the small
            leaves, so the round no longer re-reads gigabytes to fold them.
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

            # ONE forward pass plus the state blend, always the full batch,
            # always the same shape - eager, compiled, or a replay of the
            # captured graph, depending on STEP_MODE.
            preds = self._step_fn()

            # Immediately, and before anything else runs: in the graph modes
            # preds is the same device tensor every round, so its value has to
            # be read before the next replay overwrites it. .cpu() also
            # synchronizes the stream, which is what makes it safe for the next
            # round to overwrite the staging buffers the async copies read
            # from. A single D2H copy for the whole (num_symbols, 4) block; the
            # active rows are then picked out by index on the host.
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

        A leaf where `new` is literally `old` is already done and is skipped:
        that is how the three big state tensors come back now, written in place
        by their own cells under this same mask so the value never has to be
        streamed out and read back in again. What is left for this walk is the
        small leaves - conv windows, RGLRU and sLSTM state, the RetNet offset
        counter - which are cheap enough that the extra pass does not matter.

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
            # Same object on both sides: the cell already wrote this leaf in
            # place, under this same mask, so there is nothing left to fold.
            if new is old:
                return
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