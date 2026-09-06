# Optimization record

What was done to `NnInferenceClient` in `example_model.py`, in the order it was done, with the numbers
that justified each step. Everything measured here is an RTX A6000 (48 GB, 768 GB/s, sm_86) with torch
2.8.0 and Triton 3.4.0, 39 symbols, `tiny.parquet`. The event box is an H100, where the balance shifts;
see "What is left".

The model is fixed: four towers (xLSTM, Mamba2, RetNet, Hawk), 12 blocks each, hidden 2048, about 2B
parameters, one recurrent state per symbol. Only the client and the way the cells write their state
changed. The checkpoint is loaded and used unmodified.

## The client as it stands

One forward pass per round, over every row, always the same shape and the same addresses.

- **Rounds.** Round k takes request index k from every symbol whose queue is at least k+1 long, so a
  symbol appears at most once per round and every request gets exactly one prediction, in order.
- **Whole batch, plus a mask.** A round runs all rows; a symbol with nothing queued gets a row of zeros
  and `False` in a `(capacity,)` bool mask. No gather, no scatter, one shape forever. `num_symbols` is a
  row capacity, not a universe; see "Symbol rows" below.
- **In-place masked state writes.** The three large state leaves (the xLSTM mLSTM cell, the Mamba2
  `ssm_state`, the RetNet `recurrent_state`, about 98% of the state) are updated in place by their own
  cells under that mask and handed back as the same object. `_blend_state` skips them by identity and
  folds only the small leaves.
- **Fused Triton kernels.** Each of those updates is one kernel: load a state element once, do the
  decay, the rank-1 update, the contraction and the masked write in registers, store once. An inactive
  row is neither loaded nor stored.
- **Capture, and fp16 weights.** The forward plus the blend is `self._step_fn()`, captured once as a
  CUDA graph; every buffer it touches is allocated in `__init__` and never rebound. Every `nn.Linear`
  except `dt_proj`, `igate_proj`, `fgate_proj` and `output_proj` becomes a `CastLinear` holding fp16
  weights, casting back to fp32 on exit, so module boundaries still see fp32.

## Where the time went

Evaluator batch latency on `tiny.parquet`, 39 symbols. The latency is per batch and covers every round
in that batch, so it only compares across runs at the same `--batch-size`.

| Step | Commit | Batch latency | Answered | Max abs error per tower |
| --- | --- | --- | --- | --- |
| Original horizontal batching | `6dde31a` | 218 ms | 58% | 0.016 / 0.040 / 0.077 / 0.002 |
| Requests processed in rounds | `3205b1d` | 512 ms | 100% | about 1e-4 |
| bf16, then fp16 linear weights | `110632c`, `88496f5` | 512 ms | 100% | about 1e-4 |
| All rows every round, mask instead of gather/scatter | `ae081bd` | 732 ms | 100% | about 1e-4 |
| Manual CUDA graph | `3f01260` | 546 ms | 100% | about 1e-4 |
| torch.compile reduce-overhead | `3f01260` | 223 ms | 100% | about 1e-4 |
| In-place masked cell writes, sum-form contractions | `565c29c` | 202 ms | 100% | about 1e-4 |
| Fused Triton kernels with row gating | `548091d` | 96 ms | 100% | 1e-4 to 3e-4 |

Error columns are XLSTM / Mamba2 / RetNet / Hawk, maximum absolute error against the parquet's targets,
whose typical magnitude is 0.02 to 0.09. Two rows go the wrong way, both deliberately:

- 218 to 512 ms is not a regression. The original code processed one request per symbol per batch and
  silently dropped the rest, so it answered 58% of requests and advanced each symbol's state past events
  it never predicted, which is where those 0.016 to 0.077 errors came from. The round loop answers all
  of them, at more forward passes per batch.
- 512 to 732 ms is the price of a fixed shape: running all 39 rows when a few are active is a loss in
  eager and the precondition for capturing the pass as a CUDA graph. The next two rows collect on that,
  and the Triton kernels then make idle rows nearly free.

## Diagnosis

**State-bound, not weight-bound.** The state is 146 MB per symbol row, 5.6 GB at 39 rows:

| Tower | Per row | Share |
| --- | --- | --- |
| xLSTM mLSTM cell, 12 blocks of (8, 512, 512) fp32 | 96 MB | 66% |
| Mamba2 `ssm_state`, 12 blocks of (64, 64, 128) fp32 | 24 MB | 17% |
| RetNet `recurrent_state`, 12 blocks of (8, 256, 256) fp32 | 24 MB | 17% |
| Everything else (conv windows, sLSTM, RGLRU, counters) | about 2 MB | under 2% |

The weights are 3.9 GB in fp16 and are read once per round however many rows ride along. The state is
read and written per active row. Before the kernel work the mLSTM cell made four passes over its state
per block per round (the update, the read back for the contraction, then the blend's read and write),
which at 39 rows is 23 ms of a 42 ms round for that one leaf. So the work went into the state path
first and into precision second.

**Why autocast would have been wrong.** `torch.autocast` keeps master weights in fp32 and casts them on
every pass: it reads the full fp32 tensor and writes a bf16 copy, which is more HBM traffic than reading
fp32, not less. Storing the weights in fp16 halves the bytes actually streamed. The cast back to fp32 at
each module boundary is load-bearing: the `CausalConv1d` convolution, `BlockLinear`'s einsum, the RetNet
rotary built off an int32 counter, Hawk's `sqrt(1 - a**2)` and the running sums in Mamba2 and xLSTM
either reject a mixed-dtype operand or drift once a long-lived state accumulates in 8 mantissa bits. The
four projections that feed exponential decays, gate rescaling or the final prediction stay fp32, and at
0.2% of the parameters cost no measurable bandwidth. fp16 and bf16 measured the same on bandwidth; fp16
won on error against `tiny.parquet` and is the default.

**Why the einsum blocked fusion.** Each of the three cells computed its output as
`torch.einsum("bnh,bnhk->bnk", q, state_new)`. einsum lowers to a bmm, an extern cuBLAS call, and an
extern call fuses with nothing: the new state is materialized to HBM, read back by cuBLAS, then read and
rewritten by the blend. Written instead as a broadcast multiply plus `.sum(dim=2)` it is a reduction
inductor owns, so the update, the contraction and the masked write-back can fuse. That took the mLSTM
cell from six passes to four, the 202 ms row; the Triton kernels then took it to one read and one write.
Mind the axis: both trailing axes are `head_size`, so contracting dim 3 instead of dim 2 has the right
shape and the wrong value. It is also not pure reassociation: the client sets `matmul_precision("high")`
and `allow_tf32`, so the bmm ran that 256-long contraction in TF32 (10 mantissa bits), while a multiply
plus a reduction is not a matmul and neither flag reaches it, so it now runs in full fp32. The RetNet
tower's error moved by about that much, downwards.

**Why row gating matters on the live server.** The server sends about 400 requests per second over 34 to
39 symbols. A round takes tens of milliseconds, so a typical first round carries roughly 8 requests, and
later rounds carry only the symbols that queued twice, fewer still. Fusion alone does not help those
rounds: it still moves every row's state across HBM. Gating on the mask inside the kernel means an idle
row costs no traffic at all, so a thin round is cheap and spare capacity is close to free. That is the
argument for a capacity rather than a count.

## Per-round GPU time

`profile_step.py`, inductor mode, one round of `process_batch` including the H2D staging and the D2H of
the predictions.

| Configuration | Before the kernels | After the kernels |
| --- | --- | --- |
| 39 rows, all active | 43.6 ms | 31.4 ms |
| 20 rows, all active | 29.0 ms | not measured |
| 39 rows, 8 active | about 43 ms (row count, not activity) | 17.2 ms |

Before the kernels the round fit `14 ms + 0.77 ms per row`: about 14 ms of fixed cost and a per-row term
that did not care whether the row was active. After them the per-row term follows the mask. Breakdown of
the 17.2 ms round at 8 of 39 active:

| Component | Time |
| --- | --- |
| fp16 weight matmuls | about 8.6 ms |
| The three fused state kernels, 36 launches | 3.7 ms |
| Everything else (norms, convs, gates, small elementwise, staging, D2H) | the rest |

8.6 ms for 3.9 GB of weights is about 450 GB/s against the card's 768 GB/s peak, at a batch where most
of the matmuls are thin.

## Kernel benchmark

`check_kernels.py --bench`, per call, one block's worth of state. GB/s counts two passes over the active
rows only (one read, one write), and charges the reference the traffic it should have moved rather than
the traffic it did.

| Op | Rows | Active | Triton | Implied bandwidth |
| --- | --- | --- | --- | --- |
| mLSTM step | 39 | 39 | 0.997 ms | 656 GB/s |
| mLSTM step | 39 | 8 | 0.216 ms | about 620 GB/s |
| RetNet step | 39 | 39 | 0.25 ms | |
| Mamba2 step | 39 | 39 | 0.25 ms | |

That is 85% of peak bandwidth on the shape that matters, so there is nothing left in the kernels
themselves. Tiles are fixed at 32 x 128 fp32 with 4 warps and 2 stages rather than autotuned, because
autotune re-launches with several configs to choose one and that does not survive CUDA graph capture;
those five constants at the top of `model/kernels.py` are the whole tuning surface.

## Accuracy

Unchanged since the round fix: maximum absolute error 1e-4 to 3e-4 per tower against targets whose
typical magnitude is 0.02 to 0.09. fp16 weights, the sum-form contractions and the Triton kernels each
moved it within that band and none moved it out. Two checkers exist and both should pass before
anything ships:

- `check_cell_rewrite.py` materializes `model/` at the pre-rewrite commit as `oldmodel`, gives both
  sides the same weights and steps them side by side under a random mask: state agrees leaf by leaf and
  row by row, exactly the three 4-D leaves come back by identity, a masked-off row is bit-identical,
  `mask=None` writes nothing in place. Run it both ways; with `--no-kernels` a failure is the rewrite,
  without it a failure is the kernels.
- `check_kernels.py` compares each Triton kernel against its reference under mixed, all-true, all-false
  and absent masks, at every shape this repo or the live model produces plus a ragged one that exercises
  partial tiles, with strided inputs and under `torch.compile`, and checks that the wrapper refuses
  rather than repairs a bad state.

## Environment switches

| Variable | Values | Default | Effect |
| --- | --- | --- | --- |
| `LINEAR_DTYPE` | `fp16`, `bf16`, `fp32` | `fp16` | Storage dtype for the linear weights. `fp32` leaves them alone. Ignored off CUDA. The four sensitive projections stay fp32 regardless. |
| `STEP_MODE` | `inductor`, `cudagraph`, `compile`, `eager` | `inductor` | How the step runs. `inductor` is torch.compile reduce-overhead, fused kernels replayed by cudagraph trees. `cudagraph` is a manual capture of the eager kernels. `compile` is torch.compile with no graphs, for profiling, since kineto then names each fused kernel. `eager` is the reference path and the only one that runs off CUDA. |
| `CELL_KERNELS` | `triton`, anything else | `triton` | Anything but `triton` forces the pure-torch reference path in the cells. The wrappers also fall back on their own when Triton is missing or the state is not on CUDA. |

An unrecognized value for the first two prints a warning and falls back to the default.

## Commands

```bash
# Accuracy and batch latency. Use a batch size large enough that a symbol appears more than once per
# batch, and compare runs only at the same --batch-size.
python local_evaluator.py --requests-parquet-file tiny.parquet --batch-size 64 --token hf_xxx

# One round, timed and attributed. --step-mode eager (the default here) is the only mode whose
# per-label rows are reliable; compile names the fused kernels; inductor is wall clock only.
python profile_step.py --rounds 20
python profile_step.py --active 8 --rounds 20
python profile_step.py --step-mode compile --rounds 20
python profile_step.py --step-mode inductor --num-symbols 128 --rounds 20

# The rewrite, both sides of it, and the kernels.
python check_cell_rewrite.py --device cuda
python check_cell_rewrite.py --device cuda --no-kernels
python check_kernels.py --device cuda --bench
CELL_KERNELS=torch python check_kernels.py

# The live client.
python example_model.py --host $SERVER_HOST --port 8001 --num-symbols 64
```

## Box setup

The root partition on the event boxes fills, and every cache torch wants defaults to `$HOME`. Point all
of them at the large volume before the first run:

```bash
export TMPDIR=/data/tmp
export PIP_CACHE_DIR=/data/cache/pip
export HF_HOME=/data/cache/hf
export TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor
export TRITON_CACHE_DIR=/data/cache/triton
export TRITON_LIBCUDA_PATH=/data/lib          # a DIRECTORY holding libcuda.so
```

`TRITON_LIBCUDA_PATH` is the one that catches people: Triton appends the filename itself, so it wants a
directory containing a `libcuda.so` symlink, not a path to the library. If the driver library only
exists as `libcuda.so.1`, make the link first:

```bash
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /data/lib/libcuda.so
```

Without it the Triton path fails at launch and the cells fall back to torch, which costs about a third
of the round and produces no error anyone will notice in a leaderboard run.

## Cold start

| Mode | First start | Warm inductor cache | Per-round cost |
| --- | --- | --- | --- |
| `inductor` (default) | about 95 s | about 18 s | baseline |
| `cudagraph` | about 1 s | about 1 s | about 2x baseline |
| `eager` | immediate | immediate | far worse |

Warm the cache before an event: run the client, or `profile_step.py`, once on the box with the same
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` and the same `num_symbols` the real run will use. The
compile is keyed by shape, so warming at 39 rows does nothing for a run at 64. If a run has to start
cold under time pressure and cannot afford 95 s of silence, `STEP_MODE=cudagraph` starts in about a
second and gives up roughly half the throughput; switch back once the cache is warm.

## Symbol rows: `num_symbols` is a capacity

`num_symbols` sizes the state, the feature buffer and the mask. It does not name the symbols.
`symbol_to_idx` starts empty, `_row_for` hands out the next free row the first time a name arrives, and
a symbol keeps that row, and its recurrent state, for the life of the process. Assigning a row is
host-side bookkeeping only: the row still holds what `init_state` built, its mask bit stays false until
a round sets it, no device buffer changes shape, contents or address, and the captured graph stays
valid. The default capacity is 64 rows.

This replaced `{f"SYM_{i:03d}": i for i in range(num_symbols)}`, which assumed both the naming and the
size of the live universe. Any name outside that guess, an unknown string or an index at or past the
capacity, raised `KeyError` inside `process_batch`; `BaseInferenceClient._process_loop` catches it,
prints "Process error" and sleeps 100 ms, so the whole batch was lost, including the symbols that were
fine, and any round that had already run had advanced their state without answering. The universe is not
known for certain: the datasets here hold 34 to 39 symbols named `SYM_000` to `SYM_038`, and the winning
client of the 2025 event ran 20 rows without crashing, which says nothing about what the server sends.

If more distinct symbols arrive than there are rows, the ones with no row are answered with zeros and
staged nowhere, and a warning prints once per process rather than once per request. That is wrong for
those symbols and keeps the rest of the batch alive, the better of the two failures; set the capacity
above any plausible universe instead. Spare rows are nearly free in time (17.2 ms at 8 of 39 active
against 31.4 ms at 39 of 39, and the fixed per-round cost is the weight matmuls, which do not care about
the row count) and cost 146 MB of state each: 64 rows is 9.3 GB of state plus 3.9 GB of weights, 128
rows is 18.7 GB plus 3.9 GB, both comfortable on a 48 GB A6000 and on an 80 GB H100.

## What is left

The weights are the floor now: about 8.6 ms per round on the A6000, roughly 3x less on an H100. Three
things could still be taken, in descending order of what they are worth here.

- **8-bit weight storage with a dequantizing matmul.** Halves the 8.6 ms again, and it is the only
  remaining large block. It costs accuracy, and accuracy is a multiplier in the score, so it is probably
  not worth it. If it is tried, measure the score, not the latency.
- **Four towers on parallel streams.** The towers are independent until `output_proj`, so their kernels
  could overlap and hide the per-kernel gaps. On the A6000 the round is close to bandwidth bound and
  there is little gap to hide; on an H100, where the matmuls are about a third of the cost, the gaps are
  a larger share and this matters more. It interacts with CUDA graph capture, which has to record the
  side streams too.
- **Fusing the remaining small kernels.** Norms, convolutions, gates and elementwise work are the
  balance of the round after the matmuls and the state kernels. Inductor already fuses much of it; what
  is left is a long tail of small launches inside the graph, so the win is real but bounded.

The state kernels themselves are done: one read and one write per active element, nothing for an idle
one, at 85% of the card's bandwidth.
