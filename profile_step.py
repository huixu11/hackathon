#!/usr/bin/env python3
"""Where does one round of NnInferenceClient.process_batch spend its GPU time?

A "round" is one MultiTowerModel forward over all num_symbols rows plus the
blend that folds the active rows back into batched_state. This builds a batch
that is exactly one round (one queued request per symbol, so process_batch's
round loop runs once), times it with CUDA events, then profiles it, attributing
device time to each of the four towers and to the blend via record_function.
Measures only - no accuracy, no server. Run from the repo root.

    python profile_step.py                            # eager, all symbols active
    python profile_step.py --active 8 --rounds 20     # 8 active, rest masked off
    python profile_step.py --step-mode compile --rounds 20     # fused kernels, by name
    python profile_step.py --step-mode cudagraph --rounds 20   # wall clock only
    python profile_step.py --num-symbols 128 --rounds 20
    python profile_step.py --requests-parquet-file small.parquet --token hf_xxx

The per-label rows at the bottom are only reliable under --step-mode eager:
those record_function ranges sit inside the region torch.compile traces, and
dynamo is free to drop a profiler context it traces through, in which case the
rows print "n/a (range never entered)". What --step-mode compile buys is the
table above them - with no cudagraphs each of inductor's fused kernels is
launched on its own, so kineto names them one at a time. cudagraph and inductor
both collapse the whole step into a single launch, so under those only the wall
clock is real.

Nothing here writes to the repo and nothing here touches example_model.py or
model/ on disk; the record_function labels are installed on the live objects,
after the timing loop, and only for the profiled calls.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The insert local_evaluator.py does, plus the repo root itself so the imports
# still resolve when this is run from somewhere other than the repo root. The
# repo root goes in last so it ends up first and wins.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import PendingRequest  # noqa: E402
from example_model import STEP_MODES, NnInferenceClient  # noqa: E402

# In the order MultiTowerModel.__init__ builds self.towers.
TOWER_LABELS = ["tower:xlstm", "tower:mamba2", "tower:retnet", "tower:hawk"]
BLEND_LABEL = "blend"
WARMUP_CALLS = 3
# Profiling every round of a 20-round run buys nothing and costs a lot of
# memory: one round is already tens of thousands of kineto events.
DEFAULT_PROFILE_ROUNDS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--requests-parquet-file", type=str, default="tiny.parquet")
    p.add_argument("--num-symbols", type=int, default=None,
                   help="Batch rows. Default: from the parquet, max index + 1.")
    p.add_argument("--active", type=int, default=None,
                   help="Symbols given a request each round. Default: all.")
    p.add_argument("--rounds", type=int, default=5, help="Timed calls.")
    p.add_argument("--profile-rounds", type=int, default=None,
                   help=f"Profiled calls. Default: min(--rounds, {DEFAULT_PROFILE_ROUNDS}).")
    p.add_argument("--step-mode", type=str, default="eager", choices=STEP_MODES,
                   help="Written to STEP_MODE before the client reads it. The "
                        "client's own default is blocks; this defaults to "
                        "eager, the only mode the per-label attribution is "
                        "reliable in. compile names the fused kernels instead.")
    p.add_argument("--row-limit", type=int, default=25,
                   help="Rows of the key_averages table to print.")
    p.add_argument("--token", type=str, default=None, help="Hugging Face token.")
    return p.parse_args()


def build_round(df: pd.DataFrame, active: int,
                num_features: int) -> dict[str, list[PendingRequest]]:
    """One request per symbol for the first `active` symbols.

    The shape process_batch wants: Dict[str, List[PendingRequest]], keyed by
    "SYM_%03d" strings, with `features` a plain list of Python floats. The
    client hands a name a state row the first time it sees one, so inserting
    the names here in index order reproduces the identity mapping the fixed
    symbol_to_idx used to hardcode: SYM_007 lands on row 7.

    Every list holds exactly one request, so process_batch's round loop
    (`for k in range(max(len(reqs)))`) runs once: one forward, one blend, one
    D2H.

    The row is padded or truncated to num_features, so a parquet whose feature
    count does not match ModelConfig still produces a runnable batch. Only the
    shape matters here - nothing in this script looks at a prediction.
    """
    if len(df) == 0:
        raise SystemExit("the parquet has no rows; nothing to build a round from")
    feature_cols = [c for c in df.columns if c.startswith("feature")][:num_features]
    if not feature_cols:
        raise SystemExit("no feature* columns in the parquet")

    now = time.time()
    round_inputs: dict[str, list[PendingRequest]] = {}
    for i in range(active):
        row = df.iloc[i % len(df)]  # wrap if the parquet has fewer rows than symbols
        features = [float(row[col]) for col in feature_cols]
        if len(features) < num_features:
            features.extend([0.0] * (num_features - len(features)))
        symbol = f"SYM_{i:03d}"
        round_inputs[symbol] = [PendingRequest(
            unique_id=i,
            symbol=symbol,
            features=features,
            received_time=now,
        )]
    return round_inputs


def state_stats(state) -> tuple[int, int, int]:
    """(bytes, tensor leaves, other leaves) over a tree of lists/tuples/dicts.

    Same container zoo _blend_state walks: lists of towers, tuples per block
    (RetNet and xLSTM hand back tuples, Mamba2 and Hawk lists), and dicts,
    which nothing builds today but the blend handles. A leaf that is neither a
    container nor a tensor - a bare int, None - contributes no bytes and is
    counted separately, exactly as the blend walks past it.

    Bytes are deduped by data_ptr so an aliased leaf is not counted twice. The
    leaf count is deliberately not deduped: the blend runs at most one
    torch.where plus one copy_ per occurrence, so twice it bounds the blend's
    launches from above. It is only a bound and no longer the count, because a
    leaf its cell already wrote in place under the mask comes back by identity
    and the blend skips it.
    """
    total = tensors = others = 0
    seen: set[int] = set()
    stack = [state]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, torch.Tensor):
            tensors += 1
            ptr = node.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += node.numel() * node.element_size()
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        else:
            others += 1
    return total, tensors, others


def record_wrapper(fn, label: str):
    """`fn` under record_function(label), but only on the outermost call.

    The re-entrancy guard is for _blend_state, which recurses through
    self._blend_state: without it every state leaf would open its own "blend"
    range. key_averages() merges every event carrying that key into one row,
    and an outer range's device_time_total already includes its children's, so
    the nested copies would be summed on top of the parent and the blend would
    read several times its real cost. Towers do not recurse but share the code
    path, and the guard is a no-op for them.

    *args/**kwargs, so the wrapper carries whatever signature the wrapped
    callable had: Tower.forward's (x, state, mask=...) - the mask arrives as a
    keyword and passes straight through - and _blend_state's (old, new, mask).
    Both are wrapped as already-bound methods, so neither takes self.
    """
    inside = False

    def wrapped(*args, **kwargs):
        nonlocal inside
        if inside:
            return fn(*args, **kwargs)
        inside = True
        try:
            with torch.profiler.record_function(label):
                return fn(*args, **kwargs)
        finally:
            inside = False

    return wrapped


def install_record_functions(client) -> None:
    """Label each tower's forward and the state blend for the profiler.

    nn.Module.__call__ resolves self.forward off the instance, and
    nn.Module.__setattr__ falls through to object.__setattr__ for a value that
    is not a Parameter/Module/Tensor, so an instance attribute shadows the
    class method. client is a plain object, so the same trick works on its
    bound _blend_state.
    """
    towers = list(client.model.towers)
    if len(towers) != len(TOWER_LABELS):
        raise RuntimeError(f"expected {len(TOWER_LABELS)} towers, found {len(towers)}")

    for tower, label in zip(towers, TOWER_LABELS, strict=True):
        wrapped = record_wrapper(tower.forward, label)  # bound method in
        tower.forward = wrapped
        if tower.forward is not wrapped:
            raise RuntimeError(f"could not shadow forward on {type(tower).__name__}")

    blend = record_wrapper(client._blend_state, BLEND_LABEL)  # (old, new, mask)
    client._blend_state = blend
    if client._blend_state is not blend:
        raise RuntimeError("could not shadow _blend_state on the client")


def device_time_us(avg) -> float:
    """Total device time of a FunctionEventAvg, in microseconds.

    torch renamed cuda_time_total to device_time_total. In 2.8 FunctionEventAvg
    carries only the new name (the deprecated alias survives on FunctionEvent,
    not on the average), so read the new one first; the fallback is for an
    older torch. A record_function range's device_time_total is
    `sum(kernel durations) + sum(child.device_time_total)`, so a tower's label
    carries its whole subtree - and it is nonzero only because the profiler
    sets use_device="cuda" when ProfilerActivity.CUDA is in activities.
    """
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(avg, name, None)
        if value is not None:
            return float(value)
    return 0.0


def averages_table(averages, row_limit: int) -> str:
    """key_averages().table() sorted by device time, under whichever name works.

    torch 2.8 accepts both keys - _build_table rewrites "cuda" to "device" in
    sort_by before the getattr, so "cuda_time_total" still sorts - but
    device_time_total is the name FunctionEventAvg actually has, so ask for
    that first. cuda_time_total is the fallback for an older torch, and an
    unsorted table is better than no table at all.
    """
    problems = []
    for sort_by in ("device_time_total", "cuda_time_total", None):
        try:
            return averages.table(sort_by=sort_by, row_limit=row_limit)
        except Exception as exc:
            problems.append(f"sort_by={sort_by!r}: {type(exc).__name__}: {exc}")
    return "could not build the table (" + "; ".join(problems) + ")"


def kernel_launches(prof) -> int | None:
    """CUDA kernels launched inside the profiled region, counted once each.

    Kineto hangs each device kernel off the CPU op that launched it, so summing
    FunctionEvent.kernels over every event counts each kernel exactly once (the
    device-side events carry an empty list). Returns None if the shape of the
    event objects is not what this expects.
    """
    try:
        return sum(len(getattr(evt, "kernels", ()) or ()) for evt in prof.events())
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit(f"--rounds must be at least 1, got {args.rounds}")
    if args.profile_rounds is None:
        profile_rounds = min(args.rounds, DEFAULT_PROFILE_ROUNDS)
    elif args.profile_rounds < 1:
        raise SystemExit(f"--profile-rounds must be at least 1, got {args.profile_rounds}")
    else:
        profile_rounds = args.profile_rounds
    if not torch.cuda.is_available():  # the whole point is device time
        raise SystemExit("no CUDA device visible; nothing to profile")

    df = pd.read_parquet(args.requests_parquet_file)
    if args.num_symbols is None:
        if "symbol" not in df.columns:
            raise SystemExit("the parquet has no 'symbol' column; pass --num-symbols")
        # Exactly what local_evaluator.LocalEvaluator.__init__ does.
        num_symbols = max(int(sym[-3:]) for sym in df["symbol"].unique()) + 1
    else:
        num_symbols = args.num_symbols
    if num_symbols < 1:
        raise SystemExit(f"num_symbols must be at least 1, got {num_symbols}")
    active = num_symbols if args.active is None else args.active
    if not 1 <= active <= num_symbols:
        raise SystemExit(f"--active must be in [1, {num_symbols}], got {active}")

    # Set before the constructor runs: NnInferenceClient reads STEP_MODE in
    # _setup_step_fn, which __init__ calls last.
    os.environ["STEP_MODE"] = args.step_mode
    client = NnInferenceClient(num_symbols=num_symbols, token=args.token)

    n_features = len([c for c in df.columns if c.startswith("feature")])
    print(f"\nparquet={args.requests_parquet_file} rows={len(df)} "
          f"features={n_features} num_symbols={num_symbols} active={active} "
          f"rounds={args.rounds} profile_rounds={profile_rounds} "
          f"step_mode={args.step_mode} device={client.device}")
    if n_features != client.num_features:
        print(f"WARNING: the model wants {client.num_features} features and the "
              f"parquet has {n_features}; rows are truncated / zero-padded. The "
              f"timings are unaffected - the batch shape is fixed either way.")

    nbytes, tensors, others = state_stats(client.batched_state)
    extra = f" (+{others} non-tensor leaves)" if others else ""
    print(f"batched_state: {nbytes / 1024**3:.3f} GB over {tensors} tensors{extra}, "
          f"{nbytes / num_symbols / 1024**2:.3f} MB per symbol")
    # batched_state is a list of four tower states, in the order
    # MultiTowerModel.__init__ builds self.towers.
    for tower_state, label in zip(client.batched_state, TOWER_LABELS):
        t_bytes, t_tensors, _ = state_stats(tower_state)
        print(f"  {label:<14} {t_bytes / 1024**3:7.3f} GB  "
              f"{t_bytes / num_symbols / 1024**2:7.3f} MB/symbol  "
              f"{t_tensors:4d} tensors  ({100 * t_bytes / max(nbytes, 1):5.1f}%)")
    print(f"  the blend costs <= {2 * tensors} launches per round (one "
          f"torch.where plus one copy_ per tensor leaf); every leaf its cell "
          f"already wrote in place under the mask comes back by identity and "
          f"is skipped, so the real count is lower")
    print(f"  cuda memory: {torch.cuda.memory_allocated() / 1024**3:.3f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB reserved")

    round_inputs = build_round(df, active, client.num_features)

    for _ in range(WARMUP_CALLS):
        client.process_batch(round_inputs)
    torch.cuda.synchronize()

    # Wall clock. The events bracket the whole of process_batch, so this covers
    # the H2D staging copies and the D2H of the predictions, not just the model.
    # process_batch ends in a .cpu(), which synchronizes, so the host timer
    # closes on a finished round and the two numbers are comparable: host minus
    # stream is the launch overhead that never overlapped with the GPU.
    gpu_ms, host_ms = [], []
    for _ in range(args.rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        client.process_batch(round_inputs)
        host_ms.append((time.perf_counter() - t0) * 1000)
        end.record()
        torch.cuda.synchronize()  # both events are complete after this
        gpu_ms.append(start.elapsed_time(end))
    print(f"\nper round over {args.rounds} calls:")
    print(f"  gpu stream   mean {np.mean(gpu_ms):8.2f} ms   min {np.min(gpu_ms):8.2f} ms")
    print(f"  host wall    mean {np.mean(host_ms):8.2f} ms   min {np.min(host_ms):8.2f} ms")

    # Everything below is best-effort: if the profiler cannot run, or the
    # record_function labels cannot be installed, the numbers above still stand
    # and are the point of the script.
    print(f"\nprofiling {profile_rounds} round(s)...")
    try:
        # Installed only now, so the record_function ranges stay out of the
        # numbers above; the extra call absorbs any recompile the patch
        # triggers before the profiled calls start.
        install_record_functions(client)
        client.process_batch(round_inputs)
        torch.cuda.synchronize()

        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities,
                                    record_shapes=False) as prof:
            for _ in range(profile_rounds):
                client.process_batch(round_inputs)
            torch.cuda.synchronize()

        averages = prof.key_averages()
        print(averages_table(averages, row_limit=args.row_limit))

        launches = kernel_launches(prof)
        if launches is not None:
            print(f"\ncuda kernels launched: {launches / profile_rounds:.0f} per round")

        print("\ndevice time per round, by label:")
        labelled_us = 0.0
        for label in TOWER_LABELS + [BLEND_LABEL]:
            events = [a for a in averages if a.key == label]
            if not events:
                print(f"  {label:<14}      n/a  (range never entered)")
                continue
            total_us = sum(device_time_us(a) for a in events)
            labelled_us += total_us
            print(f"  {label:<14} {total_us / 1000 / profile_rounds:8.2f} ms")
        print(f"  {'labelled sum':<14} {labelled_us / 1000 / profile_rounds:8.2f} ms")
        print("note: reliable with --step-mode eager. These ranges sit inside the\n"
              "      region torch.compile traces, and dynamo may drop a profiler\n"
              "      context it traces through, so under compile they can come back\n"
              "      'n/a' - the fused-kernel rows in the table above are what that\n"
              "      mode is for. Under cudagraph the whole step is one graph launch,\n"
              "      and under inductor it is fused kernels replayed by cudagraph\n"
              "      trees, so there the per-label times and the per-kernel rows\n"
              "      both collapse.")
    except Exception:
        print("\nthe profiler section failed; the timings above still stand.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
