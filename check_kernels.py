#!/usr/bin/env python3
"""Prove model/kernels.py's Triton path matches its reference path.

For each of the three fused cell steps, under a mixed mask, an all-true mask, an
all-false mask and no mask at all: every row of the state agrees with the
reference after the call; a row the mask masked off is bit-identical to what it
was before the call, by torch.equal against a clone taken first rather than by
assert_close; the output has the right shape, dtype and layout, agrees with the
reference on the rows the mask selected, and is exactly zero on the rows it did
not.

The sizes are every shape either this repo or the live model actually puts
through a kernel, plus one that nothing does:
  * "small", the smallest shape in check_cell_rewrite.py's layer cases;
  * "cell"/"tower", the other shapes those cases produce - RetNet at head_size
    16, which is the minimum tile on both axes, and Mamba2 at bc_head_size 24,
    the one non-power-of-two extent anything here exercises;
  * "real", the model's own (n, h, k), at B=39 (local) and B=20 (the server);
  * "ragged", off the tile grid on both axes, which nothing legal produces.
That last one is the only case that makes a real tile partial: every legal size
is a power of two and the block is clamped to a power of two no larger than the
extent, so h=48 against BLOCK_H=32 and k=80 against BLOCK_K=128 are what
exercise the block masks.

Then non-contiguous inputs. Every read-only operand and the mask are handed in
as the odd half of an interleave whose even half is nan, so an operand the
wrapper failed to make contiguous is read as nan rather than as something
plausible. The mask matters most: .to(dtype) keeps a dense input's strides, so a
strided mask would otherwise reach the kernel packed and select the wrong rows.

Then the op under torch.compile, at the small size and again at the size that
ships (a different BLOCK_K and more than one h tile, so a different
specialization): the output matches eager, the mutated state matches eager, the
state actually advanced (if inductor missed the mutation through the kernel's
scf.if it would silently stop updating), and idle rows came through untouched.
Then the wrapper's refusals - a non-contiguous state, a non-fp32 state, a
non-fp32 operand, a short mask, a mask on the wrong device - because the state
is written in place and quietly repairing any of those would corrupt a live
model.

Finally, with --bench on CUDA: reference against Triton at the real sizes for
B=39 and B=20, with every row active and with one in five, in ms per call and in
the GB/s implied by two passes over the active rows' state - one read and one
write, the floor for an in-place update.

Usage:
    python check_kernels.py
    python check_kernels.py --device cuda --bench
    CELL_KERNELS=torch python check_kernels.py    # the fallback, self-consistent
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import kernels  # noqa: E402

FAILED: list[str] = []
PASSED = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}" + (f"\n      {detail}" if detail else ""))


def close(name, got, want, rtol=1e-4, atol=1e-5) -> None:
    # A tolerance, not equality: the kernel reassociates the contraction into a
    # tile tree and may contract a multiply-add into an fma, so even the state
    # update is only bit-identical by luck. The contract allows exactly this.
    try:
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
    except AssertionError as exc:
        report(name, False, " | ".join(str(exc).strip().splitlines()[:3]))
    else:
        report(name, True)


# ------------------------------------------------------------------ inputs
# Gates and decays in (0, 1) and states ~N(0, 1), so repeated application stays
# bounded and the benchmark can step the same state a hundred times without
# drifting into inf or nan and changing what it is timing.


def _draw(device, gen):
    """Normal and uniform samplers bound to this device and generator."""

    def rnd(*shape):
        return torch.randn(*shape, device=device, generator=gen)

    def uni(*shape):
        return torch.rand(*shape, device=device, generator=gen)

    return rnd, uni


def make_mlstm(batch, dims, device, gen):
    heads, size_h, size_k = dims
    rnd, uni = _draw(device, gen)
    # q, k (pre-scaled by the caller), v, i_gate, f_gate
    return rnd(batch, heads, size_h, size_k), (
        rnd(batch, heads, size_h), rnd(batch, heads, size_h) * size_h**-0.5,
        rnd(batch, heads, size_k), uni(batch, heads), uni(batch, heads),
    )


def make_retnet(batch, dims, device, gen):
    heads, size_h, size_k = dims
    rnd, uni = _draw(device, gen)
    # k_rope, v, q_scaled, decay (per head, shared by every row)
    return rnd(batch, heads, size_h, size_k), (
        rnd(batch, heads, size_h) * size_h**-0.5, rnd(batch, heads, size_k),
        rnd(batch, heads, size_h), uni(heads),
    )


def make_mamba2(batch, dims, device, gen):
    heads, size_h, size_c = dims
    rnd, uni = _draw(device, gen)
    # x, b, c, dt, decay
    return rnd(batch, heads, size_h, size_c), (
        rnd(batch, heads, size_h), rnd(batch, size_c), rnd(batch, size_c),
        uni(batch, heads), uni(batch, heads),
    )


# tag, triton op, reference op, maker, real (n, h, k), then the smaller cases as
# (label, batch, dims). The labelled ones are the shapes check_cell_rewrite.py's
# layer cases actually drive through these kernels, so a size that only fails
# there fails here first, with a name.
OPS = (
    ("mlstm", kernels.mlstm_step, kernels.mlstm_step_reference, make_mlstm,
     (8, 512, 512),
     (("small/cell", 6, (4, 32, 32)),        # XLSTM(64, mlstm_num_heads=4)
      ("ragged", 5, (3, 48, 80)))),
    ("retnet", kernels.retnet_step, kernels.retnet_step_reference, make_retnet,
     (8, 256, 256),
     (("small", 6, (4, 32, 32)),
      ("cell", 6, (4, 16, 16)),              # RetNet(64, num_heads=4): min tile
      ("ragged", 5, (3, 48, 80)))),
    ("mamba2", kernels.mamba2_step, kernels.mamba2_step_reference, make_mamba2,
     (64, 64, 128),
     (("small", 6, (8, 16, 32)),
      ("cell", 6, (8, 16, 24)),              # Mamba2(64, 16, 24): c not a power of 2
      ("tower", 6, (2, 64, 128)),            # Mamba2(64) inside MultiTowerModel
      ("ragged", 5, (5, 40, 96)))),
)


def masks(batch, device, gen):
    mixed = torch.rand(batch, device=device, generator=gen) < 0.5
    mixed[0] = True  # at least one of each, whatever the draw was
    mixed[-1] = False
    yield "mixed", mixed
    yield "all-true", torch.ones(batch, dtype=torch.bool, device=device)
    yield "all-false", torch.zeros(batch, dtype=torch.bool, device=device)
    yield "no-mask", None


# ----------------------------------------------------------------- checks


def compare(tag, fused, ref, maker, batch, dims, device, gen, note=""):
    state0, rest = maker(batch, dims, device, gen)
    for name, mask in masks(batch, device, gen):
        label = f"{tag} {note} B={batch} dims={dims} {name}".replace("  ", " ")
        want_state, got_state = state0.clone(), state0.clone()
        want = ref(want_state, *rest, mask)
        got = fused(got_state, *rest, mask)

        all_live = torch.ones(batch, dtype=torch.bool, device=device)
        live = all_live if mask is None else mask
        idle = ~live
        untouched = torch.equal(got_state[idle], state0[idle])
        report(f"{label}: idle rows of the state untouched", untouched)
        # The cells read this straight into a broadcast add or a norm, so a
        # wrong dtype or a non-contiguous result is a bug even when the numbers
        # agree; and torch.empty means a shape check is not a formality.
        shaped = (got.shape == want.shape and got.dtype == torch.float32
                  and got.device == state0.device and got.is_contiguous())
        report(f"{label}: output is a contiguous fp32 {tuple(want.shape)}", shaped,
               f"got {tuple(got.shape)} {got.dtype} on {got.device}, "
               f"contiguous={got.is_contiguous()}")
        close(f"{label}: state, all rows", got_state, want_state)
        close(f"{label}: output, active rows", got[live], want[live])
        zeroed = bool(torch.all(got[idle] == 0.0))
        report(f"{label}: output, idle rows are zero", zeroed)


def noncontiguous(tag, fused, ref, maker, batch, dims, device, gen):
    """Every read-only operand and the mask handed in as a strided view.

    The wrappers promise .contiguous() on the small operands and on the mask -
    the state is the one thing they must never copy. Interleaving each operand
    with nan makes the failure loud rather than plausible: a kernel reading the
    view as if it were packed picks up the nan sitting next to every value. The
    mask is the one that matters most, because .to(dtype) preserves a dense
    input's strides, so a strided mask that reached the kernel unpacked would
    select the wrong rows rather than raise.
    """
    state0, rest = maker(batch, dims, device, gen)
    strided = []
    for t in rest:
        wide = torch.stack([torch.full_like(t, float("nan")), t], dim=-1)
        strided.append(wide[..., 1])

    mask = torch.rand(batch, device=device, generator=gen) < 0.5
    mask[0], mask[-1] = True, False
    wide_mask = torch.zeros(batch, 2, dtype=torch.bool, device=device)
    wide_mask[:, 1] = mask
    strided_mask = wide_mask[:, 1]

    label = f"{tag} B={batch} dims={dims} strided operands"
    assert not strided_mask.is_contiguous()
    assert all(not t.is_contiguous() for t in strided)

    want_state, got_state = state0.clone(), state0.clone()
    want = ref(want_state, *rest, mask)
    got = fused(got_state, *strided, strided_mask)
    close(f"{label}: state", got_state, want_state)
    close(f"{label}: output, active rows", got[mask], want[mask])
    report(f"{label}: output, idle rows are zero",
           bool(torch.all(got[~mask] == 0.0)))
    report(f"{label}: idle rows of the state untouched",
           torch.equal(got_state[~mask], state0[~mask]))


def compile_smoke(tag, fused, maker, batch, dims, device, gen, note=""):
    state0, rest = maker(batch, dims, device, gen)
    mask = torch.zeros(batch, dtype=torch.bool, device=device)
    mask[::2] = True
    tag = f"{tag} {note}".strip()

    def step(state, mask, *args):
        # An epilogue on the output, so inductor has something to fuse around
        # the custom kernel rather than lowering to a bare call.
        return fused(state, *args, mask).mul(2.0).add(1.0)

    eager_state, comp_state = state0.clone(), state0.clone()
    want = step(eager_state, mask, *rest)
    try:
        got = torch.compile(step, fullgraph=False)(comp_state, mask, *rest)
    except Exception as exc:  # a compile failure is a real failure here
        report(f"{tag} torch.compile runs", False, f"{type(exc).__name__}: {exc}")
        return
    report(f"{tag} torch.compile runs", True)
    close(f"{tag} torch.compile output matches eager", got, want, 1e-4, 1e-4)
    close(f"{tag} torch.compile state matches eager", comp_state, eager_state,
          1e-4, 1e-4)
    # If inductor failed to see the store through the kernel's scf.if it could
    # hand the kernel a scratch buffer, and the state would stop advancing in
    # silence. Name that failure directly rather than reading it out of a
    # tolerance mismatch above.
    advanced = not torch.equal(comp_state[mask], state0[mask])
    report(f"{tag} torch.compile advanced the active rows", advanced)
    idle = ~mask
    report(f"{tag} torch.compile left idle rows untouched",
           torch.equal(comp_state[idle], state0[idle]))


def guards(tag, fused, maker, batch, dims, device, gen):
    """The wrapper must refuse, not repair, a state it writes in place."""
    state0, rest = maker(batch, dims, device, gen)
    live = torch.ones(batch, dtype=torch.bool, device=device)
    bad_first = (rest[0].to(torch.bfloat16),) + tuple(rest[1:])
    cases = (
        ("a non-contiguous state", state0.transpose(2, 3), rest, live),
        ("a non-fp32 state", state0.to(torch.float64), rest, live),
        ("a non-fp32 operand", state0.clone(), bad_first, live),
        ("a short mask", state0.clone(), rest, live[:1]),
        # A host mask against a device state would hand the kernel a pointer it
        # cannot dereference; nothing in this repo builds one, which is exactly
        # why it should raise rather than be discovered on the box.
        ("a mask on the wrong device", state0.clone(), rest, live.cpu()),
    )
    for name, state, args, mask in cases:
        try:
            fused(state, *args, mask)
        except AssertionError:
            report(f"{tag} refuses {name}", True)
        else:
            report(f"{tag} refuses {name}", False, "no AssertionError was raised")


# -------------------------------------------------------------- benchmark


def timed(call, iters=20):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def bench(device, gen):
    print(f"\n{'op':<8}{'B':>4}{'active':>8}{'ref ms':>10}{'triton ms':>11}"
          f"{'speedup':>9}{'ref GB/s':>10}{'triton GB/s':>13}")
    for tag, fused, ref, maker, dims, _ in OPS:
        for batch in (39, 20):
            state0, rest = maker(batch, dims, device, gen)
            row_bytes = state0[0].numel() * state0.element_size()
            for frac in (1.0, 0.2):
                live = max(1, round(batch * frac))
                mask = torch.zeros(batch, dtype=torch.bool, device=device)
                mask[:live] = True
                # Two passes over the active rows only: the read and the write an
                # in-place update cannot avoid. The reference is charged the same
                # traffic it should have moved, not the traffic it did, so its
                # number reads low exactly where the idle-row skip is the win.
                gb = 2.0 * live * row_bytes / 1e9
                s_ref, s_tri = state0.clone(), state0.clone()
                t_ref = timed(lambda: ref(s_ref, *rest, mask))
                t_tri = timed(lambda: fused(s_tri, *rest, mask))
                print(f"{tag:<8}{batch:>4}{f'{live}/{batch}':>8}{t_ref:>10.3f}"
                      f"{t_tri:>11.3f}{t_ref / t_tri:>8.2f}x"
                      f"{gb / (t_ref * 1e-3):>10.0f}{gb / (t_tri * 1e-3):>13.0f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    fused_live = kernels.ENABLED and device.type == "cuda"
    print(f"triton AVAILABLE={kernels.AVAILABLE} ENABLED={kernels.ENABLED} "
          f"device={device}")
    if not fused_live:
        print("NOTE: the Triton path is inactive here, so every op below falls back\n"
              "      to the reference and this run checks plumbing, not kernels.")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    for tag, fused, ref, maker, real, cases in OPS:
        small = cases[0][2]
        for note, batch, dims in cases:
            compare(tag, fused, ref, maker, batch, dims, device, gen, note)
        compare(tag, fused, ref, maker, 39, real, device, gen, "real")
        compare(tag, fused, ref, maker, 20, real, device, gen, "real/server")
        compile_smoke(tag, fused, maker, 6, small, device, gen, "small")
        if fused_live:
            # Again at the shipped size, where the kernel is specialized
            # differently - a larger BLOCK_K and more than one h tile. CUDA
            # only: with no Triton path there is nothing here inductor has not
            # already been asked at the small size, and compiling a 167 MB
            # reduction on CPU costs minutes to learn it.
            compile_smoke(tag, fused, maker, 20, real, device, gen, "real")
            # Only the kernel wrappers assert or promise anything about layout;
            # the reference accepts a non-contiguous state, raises a
            # RuntimeError rather than an AssertionError on a short mask, and
            # says nothing about a strided operand.
            noncontiguous(tag, fused, ref, maker, 6, small, device, gen)
            guards(tag, fused, maker, 6, small, device, gen)

    if args.bench:
        if device.type != "cuda":
            print("\n--bench needs CUDA; skipped.")
        else:
            bench(device, gen)

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  failed: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
