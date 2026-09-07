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


# Values accepted in the STEP_MODE environment variable. All five produce a
# self._step_fn() that runs the forward pass and folds the new state in; they
# differ only in how much per-round launch overhead is paid to do it, and in
# how much compilation is paid up front to get there.
#
#   inductor   torch.compile(mode="reduce-overhead"): inductor's kernels,
#              replayed by cudagraph trees.
#   blocks     the same fused kernels as inductor, reached in a fraction of the
#              compile time, and replayed from a graph captured by hand.
#              torch.compile is applied to each Block instead of to the whole
#              step: dynamo caches compiled code against the code object it
#              traced, and all 48 blocks share one Block.forward, so the
#              compiler runs once per distinct trace - four, one per tower's
#              layer type - and the other 44 blocks are served by the entry
#              their variant already produced. (Per-Block torch.compile with
#              cudagraphs pinned off, so nothing tries to capture inside the
#              capture, plus fullgraph=True and dynamic=False; a hand-built
#              CUDA graph of the whole step then goes over the top, exactly as
#              in "cudagraph", so a round is still one replay.) Compiling
#              self._step instead hands inductor one graph covering all 48
#              blocks, which is where the ~95 s of "inductor" goes.
#              Expect the per-round cost to land near "inductor" rather than on
#              it: the block bodies are the same fused kernels, but what sits
#              between the blocks stays eager here - each tower's input_up /
#              input_down and output_proj, the concat, and the ~180 small state
#              leaves _blend_state folds. Those are inside the graph, so they
#              cost launches on the device and nothing on the host, but
#              "inductor" gets to fuse them and this does not.
#   cudagraph  the eager kernels, captured once by hand into a CUDA graph.
#   compile    torch.compile with the default mode: inductor's kernels, but no
#              CUDA graphs, so each one is launched on its own. Slower than
#              inductor and meant for profiling - the profiler sees the fused
#              kernels individually, by name, instead of one graph launch.
#   eager      no capture at all - the reference path, and the only one that
#              runs anywhere but CUDA.
STEP_MODES = ("inductor", "blocks", "cudagraph", "compile", "eager")
DEFAULT_STEP_MODE = "blocks"

# Enough passes to settle whatever is autotuned before the graph is recorded:
# inductor's kernel selection, cudnn.benchmark's algorithm search, and the
# allocator blocks the capture will bake in.
WARMUP_STEPS = 3


def dynamo_variant_count(fn) -> int | None:
    """How many compiled variants dynamo holds for `fn`'s code object.

    The whole premise of STEP_MODE=blocks is that this number is the number of
    tower types and not the number of blocks: dynamo keys its cache on the code
    object, and with config.inline_inbuilt_nn_modules on (the torch 2.8 default)
    a module's parameters enter the graph as inputs guarded by tensor properties
    rather than by id, so one entry serves every Block instance whose layer has
    the same type. Reading the count back is the only cheap way to see that it
    actually happened - the alternative failure is silent, because past
    config.recompile_limit dynamo stops compiling the frame and runs it eager,
    which here would be an eager block recorded into the graph at full speed
    cost and no error.

    Private API, so None rather than an exception if it moves: the caller prints
    what it gets and nothing depends on the answer.
    """
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        return len(_debug_get_cache_entry_list(fn))
    except Exception:
        return None


class NnInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        device: str | None = None,
        token: str | None = None,
    ):
        """Build the model and the fixed-shape buffers one round runs against.

        `num_symbols` is a ROW CAPACITY, not a universe. It sizes the batched
        state and the two device buffers - that is, how many symbols can be
        tracked at once - and nothing here decides which names those rows
        belong to. The mapping starts empty and process_batch fills it in on
        first sight of a name, so the client works against any naming and any
        universe size at or under the capacity. The evaluator still passes this
        as num_symbols, so the name stays.

        Rows past the live universe cost memory and almost nothing else: a row
        no symbol owns is never marked active, the state kernels skip inactive
        rows, and the per-round cost is dominated by streaming the weights,
        which happens once however many rows ride along. So the capacity is
        meant to be set with headroom - about 146 MB of GPU state per row.
        """
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

        # How many independent recurrent states the device holds. Every buffer
        # below is this tall, and it never changes for the life of the process.
        self.capacity = num_symbols

        # Initialize a SINGLE batched state covering every row, owned or not.
        self.batched_state = self.model.init_state(num_symbols, self.device)

        # Symbol -> row in that batch. EMPTY on purpose: rows are handed out by
        # _row_for on first sight of a name, and never taken back. Seeding it
        # with {f"SYM_{i:03d}": i} instead assumed both the naming and the size
        # of the live universe, and any name outside that guess - an unknown
        # string, or an index at or past the capacity - raised a KeyError in
        # the middle of process_batch, which the client loop catches and logs
        # after the whole batch of requests is already lost.
        self.symbol_to_idx: Dict[str, int] = {}

        # Width of one prediction. MultiTowerModel.forward runs its one
        # output_proj (hidden_size -> 1) over each tower's output and
        # concatenates the results along dim 1, so preds is one column per
        # tower and nothing else. Only the overflow path needs this, to size
        # the zeros it answers with; every other prediction takes its width
        # from the preds the model actually returned.
        self.num_outputs = len(self.model.towers)

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

        print(f"Ready: capacity {num_symbols} rows, symbols assigned on first sight")

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

    def _capture_step(self) -> None:
        """Record self._step into a CUDA graph and bind _step_fn to replaying it.

        Shared by the two modes that build the graph by hand - "cudagraph",
        which records the eager kernels, and "blocks", which records whatever
        the per-Block compiles left behind. What is recorded differs; the
        recording does not, so it lives in one place and the two cannot drift.

        Whatever runs inside has to be fully warmed by the time capture starts:
        capture records launches, not the Python that decides them, so a first
        call that compiles, autotunes or allocates would bake that one call's
        choices - or nothing at all - into the graph. WARMUP_STEPS passes with
        an all-false mask do that warming, and leave batched_state untouched.
        The passes have to be the ones the capture will record, so a mode whose
        first call also COMPILES has to pay that call before getting here, or it
        spends one of its three settled passes on the compiler; "blocks" does.
        """
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
            # Every mode but eager is CUDA-only here - written as "not eager"
            # rather than a list, so a mode added later is covered by default
            # and has to opt out rather than remember to opt in. The reasons:
            # mark_static_address exists to feed cudagraph trees, CUDAGraph has
            # no CPU or MPS analogue, and the warmup loops synchronize the CUDA
            # stream. "blocks" is CUDA-only for the second of those - the
            # per-Block compiles alone would run anywhere, but the capture they
            # feed would not.
            print(
                f"STEP_MODE={mode} needs CUDA; device is "
                f"{torch.device(self.device).type}, so the step runs eager"
            )
            mode = "eager"

        started = time.perf_counter()

        # Only "blocks" fills these in, and only "blocks" prints them; bound
        # here so the print below reads variables rather than maybe-unbound
        # names. n_traces stays None when the private API that reads it moves.
        n_wrapped = 0
        n_traces: int | None = None
        compile_s = 0.0

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

        elif mode == "blocks":
            # Compile the Block, not the step. Dynamo caches compiled code
            # against the code object it traced, and every one of the 48 blocks
            # is an instance of the same Block class running the same
            # Block.forward, so the compiler pays for four traces - one per
            # tower's layer type - and the remaining 44 blocks are served by
            # the cache entry their variant already filled. That is the whole
            # difference from "inductor", which hands inductor a single graph
            # spanning all 48 and takes about 95 s to schedule it.
            #
            # See the "inductor" branch above for why these imports are spelt
            # `from torch import ...`.
            from torch import _dynamo, _inductor  # noqa: F401

            # What lets one cache entry serve all twelve blocks of a tower.
            # With inlining on, a block's parameters and buffers enter the
            # graph as INPUTS guarded by tensor properties; with it off, dynamo
            # specializes the module by id and each of the 48 instances would
            # miss the cache and compile again, which is the 95 s this mode
            # exists to avoid. True is already the torch 2.8 default - set
            # explicitly so the mode does not quietly become a 48-way recompile
            # if the default (or the justknob behind it) ever moves.
            torch._dynamo.config.inline_inbuilt_nn_modules = True

            # The other way that inlining stops buying anything. Inductor's
            # freezing pass wants the parameters folded in as constants, so
            # dynamo asks for them by ADDRESS when it is on: builder.wrap_module
            # calls mark_static_input(p, guard=is_parameter_freezing()), and a
            # guarded static input is a data_ptr guard per parameter, which no
            # other block can satisfy. is_parameter_freezing() is `freezing and
            # not torch.is_grad_enabled()` and everything below runs under
            # inference_mode, so here it is just `freezing`. Off by default; the
            # global is what has to change, not the backend's options, because
            # dynamo reads it while tracing, before the options are patched in
            # around the backend call.
            if torch._inductor.config.freezing:
                print(
                    "STEP_MODE=blocks is turning inductor freezing off: it "
                    "guards every parameter by address, which would give each "
                    "of the 48 blocks its own compile"
                )
                torch._inductor.config.freezing = False

            # The recompile limit is counted per code object, and four variants
            # now live under one. The 2.8 default of 8 would fit, but there is
            # no reason to sit two entries under the cliff: past the limit
            # dynamo stops compiling that frame and silently runs it eager,
            # which here would mean an eager block captured into the graph and
            # a per-round cost quietly back where it started. max(), not a bare
            # assignment, so a caller that already raised it keeps its value.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 16
            )

            # Wrap each block in place. nn.ModuleList.__setitem__ re-registers
            # the entry, so the tower holds OptimizedModule wrappers from here
            # on; Tower.forward reaches them as block(x, block_state,
            # mask=mask) through nn.Module.__call__, which on an
            # OptimizedModule runs the compiled forward and passes the mask
            # keyword straight through.
            #
            # Wrapping is the only change to the module tree, and nothing else
            # in the process needs the tree unwrapped. The linears are already
            # CastLinear by now (convert_linears runs in __init__, well before
            # this) and are ordinary modules that trace like any other.
            # init_state is not called again either - self.batched_state was
            # built in __init__, before _setup_step_fn - and would still work
            # if it were: OptimizedModule.__getattr__ forwards an attribute it
            # does not define to the module it wraps, so Tower.init_state's
            # block.init_state(...) resolves to the real Block's.
            #
            # The class is read before the loop because after it there are no
            # plain Blocks left to ask, and dynamo_variant_count needs the
            # unwrapped Block.forward - the code object the cache is keyed on.
            block_cls = type(self.model.towers[0].blocks[0])
            for tower in self.model.towers:
                for i in range(len(tower.blocks)):
                    # fullgraph=True because a graph break inside a block would
                    # put eager Python between the fused kernels, and the
                    # capture below would record only the launches, not the
                    # Python - better to fail loudly here. Nothing in the model
                    # should break one: "inductor" traces this same code into a
                    # single graph over all 48 blocks today, Triton launches
                    # included. dynamic=False because the batch is
                    # self.capacity forever, and letting dynamo mark it dynamic
                    # would cost a recompile and give up the shape
                    # specialization for nothing.
                    #
                    # triton.cudagraphs pinned off rather than left to its
                    # default, which is `TORCHINDUCTOR_CUDAGRAPHS == "1"` and so
                    # is one environment variable away from turning on. This
                    # mode captures the step by hand; a compiled block that also
                    # ran cudagraph trees would be capturing inside that
                    # capture. Same options on every block, so all 48 backend
                    # objects still compare equal, which is what lets them share
                    # dynamo's cache entries (extra_state.cpp's backend_match
                    # falls back to ==, and _TorchCompileInductorWrapper.__eq__
                    # compares exactly this config and dynamic).
                    tower.blocks[i] = torch.compile(
                        tower.blocks[i],
                        fullgraph=True,
                        dynamic=False,
                        options={"triton.cudagraphs": False},
                    )
                    n_wrapped += 1

            # Pay the compiles here, not inside _capture_step. Dynamo traces on
            # first call, so without this pass the first of the three warmups
            # would be the one that compiles and autotunes, leaving only two
            # settled passes before the capture where the other hand-captured
            # mode gets three. An all-false mask throughout, so no state is
            # touched however long the compiler takes.
            with torch.inference_mode():
                self._stage_idle_round()
                self._step()
                torch.cuda.synchronize()

            compile_s = time.perf_counter() - started
            n_traces = dynamo_variant_count(block_cls.forward)
            if n_traces is not None and n_traces > len(self.model.towers):
                print(
                    f"WARNING: dynamo holds {n_traces} compiled variants of "
                    f"{block_cls.__name__}.forward; one per tower type "
                    f"({len(self.model.towers)}) was expected. The blocks are "
                    f"not sharing cache entries, so this mode is paying for "
                    f"them one at a time, and past "
                    f"torch._dynamo.config.cache_size_limit "
                    f"({torch._dynamo.config.cache_size_limit}) the rest run "
                    f"eager inside the captured graph."
                )

            self._capture_step()

        elif mode == "cudagraph":
            self._capture_step()

        elapsed = time.perf_counter() - started
        if mode == "blocks":
            # Split, because the two halves answer different questions: the
            # compile number is the one this mode exists to shrink, and the
            # trace count is the reason it is small.
            print(
                f"Step mode: {mode} ({n_wrapped} blocks wrapped, "
                f"{'?' if n_traces is None else n_traces} traces, "
                f"compile {compile_s:.1f} s, capture {elapsed - compile_s:.1f} s)"
            )
        else:
            print(f"Step mode: {mode} (warmup {elapsed:.1f} s)")

    # On the class, not the instance: the overflow warning prints once per
    # process, and stays true afterwards.
    _overflow_warned = False

    def _row_for(self, symbol: str) -> int | None:
        """The state row `symbol` owns, taking the next free one on first sight.

        Pure host-side bookkeeping - a dict read, and at most a dict store.
        Handing a symbol a row deliberately touches nothing on the device: the
        row still holds exactly what init_state built for it, because no round
        has ever marked it active, and its mask bit stays false until a round
        sets it. So no device buffer changes shape, contents or address when a
        new name shows up, and the captured graph stays valid.

        Rows are handed out densely and never reclaimed - a symbol keeps its
        row, and therefore its recurrent state, for the life of the process -
        so the next free row is simply how many are already taken.

        Returns None when the capacity is exhausted, meaning more distinct
        symbols have arrived than there are rows. That should not happen if the
        capacity is chosen with headroom, which is cheap to do: an unowned row
        is skipped by the state kernels and rides along in a pass whose cost is
        the weight matmuls either way. When it does happen the caller answers
        with zeros instead of raising, because a raise inside process_batch
        loses every request in the batch, not just this symbol's.
        """
        row = self.symbol_to_idx.get(symbol)
        if row is not None:
            return row

        row = len(self.symbol_to_idx)
        if row >= self.capacity:
            if not NnInferenceClient._overflow_warned:
                # Once per process. At ~400 requests a second, a print per
                # offending request would be its own outage.
                NnInferenceClient._overflow_warned = True
                print(
                    f"WARNING: out of state rows ({self.capacity} of them, all "
                    f"owned); {symbol!r} and every later new symbol are being "
                    f"answered with zeros, which is wrong but keeps the rest of "
                    f"the batch alive. Raise --num-symbols (~146 MB of GPU "
                    f"memory per row) above the size of the live universe. "
                    f"This prints once."
                )
            return None

        self.symbol_to_idx[symbol] = row
        return row

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

        Each symbol is resolved to a row once per call, through self.capacity
        rows of state. self.symbol_to_idx starts empty and _row_for fills it in
        on first sight, next free row first, so no naming or universe size is
        assumed and an unfamiliar name is not an error. A symbol keeps its row
        forever. If more distinct symbols arrive than there are rows, the ones
        with no row are answered with zeros and staged nowhere - no row, no
        mask bit, no state - and a warning prints once; with the capacity set
        with headroom that path never runs.

        Within a round the model runs over ALL self.capacity rows against the
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

        # Resolve each symbol to its row once for the whole call, rather than
        # once per round, and give a row to any name seen for the first time.
        # Fixed order, so indices and requests stay aligned within every round.
        symbol_items = []
        for symbol, reqs in requests_by_symbol.items():
            if not reqs:
                continue
            row = self._row_for(symbol)
            if row is None:
                # No row left. Answer the requests - dropping them, or letting
                # a KeyError out of here, costs the whole batch and not just
                # this symbol - but stage nothing, so the round keeps its shape
                # and no state is disturbed. _row_for has warned once already.
                for req in reqs:
                    all_unique_ids.append(req.unique_id)
                    all_predictions.append([0.0] * self.num_outputs)
                continue
            symbol_items.append((row, reqs))

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

            for row, symbol_requests in symbol_items:
                if len(symbol_requests) > k:
                    # Take this symbol's k-th queued request
                    req = symbol_requests[k]

                    batch_indices.append(row)
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
        default=64,
        help="Number of state rows to allocate - a capacity, not a symbol "
             "list. Symbols are given a row on first sight, so the naming and "
             "the size of the live universe need not be known ahead of time. "
             "Spare rows are cheap: a row no symbol owns is never active, and "
             "the state kernels skip inactive rows, so it rides along in a "
             "pass whose cost is the weight matmuls anyway. Each row costs "
             "about 146 MB of GPU memory, so set this above any plausible "
             "universe size - a symbol that arrives with no row left is "
             "answered with zeros.",
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