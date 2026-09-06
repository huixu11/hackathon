#!/usr/bin/env python3
"""Prove the masked in-place cell rewrite matches the code it replaced.

Materializes model/ at a git ref (default 3b2e25c, the pre-rewrite commit) into
a temporary directory as a package named `oldmodel`, imports it beside the
working tree's `model`, gives both the same weights, and steps them side by
side under a random per-step mask. The old path folds the state the way the
client used to - one torch.where per leaf, outside the model. The new path
hands the mask to the model and folds only the leaves that did not come back by
identity, exactly as example_model._blend_state now does.

Checked per layer type (xLSTM, Mamba2, RetNet, Hawk) and for MultiTowerModel:
  * old -> new state_dict loads with strict=True, so no key was renamed
  * outputs agree on the rows the mask selected (idle rows are garbage on both)
  * every state leaf agrees on every row, after every step
  * exactly the 4-D leaves - the xLSTM mLSTM cell, the Mamba2 ssm_state and the
    RetNet recurrent_state, the only 4-D state in this model - come back as the
    SAME tensor object, and no other leaf does
  * a row the mask masked off is bit-identical to what it was before the step
  * with mask=None nothing comes back by identity and the state handed in is
    left untouched
  * the whole-model masked step also runs inside torch.inference_mode(), and on
    CUDA under torch.compile

Weights are randomized here - several parameters ship as torch.empty and are
only ever filled by the checkpoint - so this proves the code equivalent, not
the checkpoint accurate. Run local_evaluator.py on tiny.parquet for that, with
a partially-false mask.

Usage:
    python check_cell_rewrite.py
    python check_cell_rewrite.py --ref 3b2e25c --device cuda
    python check_cell_rewrite.py --bench       # CUDA only, real sizes, B=39
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import math
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
PKG_FILES = ("__init__.py", "modules.py", "xlstm.py", "mamba2.py", "retnet.py",
             "hawk.py", "inference_model.py")
BIG_DIM = 4  # the three tensors this rewrite is about are the only 4-D leaves
OLD: dict = {}
NEW: dict = {}


def load_old(ref: str, tmp: Path) -> dict:
    """Write model/ at `ref` out as a package named oldmodel and import it."""
    pkg = tmp / "oldmodel"
    pkg.mkdir()
    for name in PKG_FILES:
        r = subprocess.run(["git", "show", f"{ref}:model/{name}"],
                           cwd=REPO, capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"git show {ref}:model/{name} failed: "
                             f"{r.stderr.decode(errors='replace').strip()}")
        (pkg / name).write_bytes(r.stdout)  # relative imports work inside a package
    sys.path.insert(0, str(tmp))
    return {f[:-3]: importlib.import_module(f"oldmodel.{f[:-3]}") for f in PKG_FILES[1:]}


def load_new() -> dict:
    sys.path.insert(0, str(REPO))
    return {f[:-3]: importlib.import_module(f"model.{f[:-3]}") for f in PKG_FILES[1:]}


# --------------------------------------------------------------- state trees

def leaves(tree):
    """Every tensor leaf, in _blend_state's walk order."""
    if isinstance(tree, (list, tuple)):
        for t in tree:
            yield from leaves(t)
    elif isinstance(tree, dict):
        for t in tree.values():
            yield from leaves(t)
    elif isinstance(tree, torch.Tensor):
        yield tree


def mask_view(mask, t):
    return mask.view(mask.shape[0], *(1,) * (t.dim() - 1))


def blend_reference(old, new, mask):
    """The fold the client did before the rewrite: a fresh tree, one where per leaf."""
    if isinstance(old, list):
        return [blend_reference(o, n, mask) for o, n in zip(old, new, strict=True)]
    if isinstance(old, tuple):
        return tuple(blend_reference(o, n, mask) for o, n in zip(old, new, strict=True))
    if isinstance(old, dict):
        return {k: blend_reference(old[k], new[k], mask) for k in old}
    return torch.where(mask_view(mask, old), new, old)


def blend_client(old, new, mask):
    """example_model._blend_state: in place, skipping a leaf that came back by identity."""
    if isinstance(old, (list, tuple)):
        for o, n in zip(old, new, strict=True):
            blend_client(o, n, mask)
    elif isinstance(old, dict):
        for k in old:
            blend_client(old[k], new[k], mask)
    elif isinstance(old, torch.Tensor):
        if new is old:
            return
        old.copy_(torch.where(mask_view(mask, old), new, old))


def clone_tree(tree):
    return [t.clone() for t in leaves(tree)]


def restore_tree(tree, snap):
    for t, s in zip(leaves(tree), snap, strict=True):
        t.copy_(s)


def state_bytes(tree) -> int:
    seen, total = set(), 0
    for t in leaves(tree):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


def random_mask(batch, dev):
    m = torch.rand(batch) < 0.5
    m[0], m[1] = True, False  # at least one active row and one idle row
    return m.to(dev)


def randomize_(module):
    """Give every parameter and buffer a well-conditioned value.

    Several ship as torch.empty and are only ever filled by the checkpoint -
    BlockLinear's weights, RetNet's decay/angle buffers, RGLRU's `a` - so an
    untouched module can hold nan and make every comparison below vacuous. The
    ranges keep the gates and decays where a trained model puts them.
    """
    mods = dict(module.named_modules())  # mods[""] is the root
    norms = ("LayerNorm", "RMSNorm", "GroupNorm")
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        if not t.is_floating_point():
            continue
        parent, _, leaf = name.rpartition(".")
        cls = type(mods[parent]).__name__
        if cls == "RGLRU" and leaf == "a":
            t.uniform_(0.1, 0.9)         # base of a ** (c * r); must stay in (0, 1)
        elif cls == "Mamba2" and leaf == "a":
            t.uniform_(-16.0, -1.0)      # exp(a * dt) is a decay, so a < 0
        elif cls == "Mamba2" and leaf == "d":
            t.uniform_(0.5, 1.5)
        elif cls == "RetNet" and leaf == "decay":
            t.uniform_(0.85, 0.99)
        elif cls == "RetNet" and leaf == "angle":
            t.uniform_(0.01, 0.2)
        elif leaf == "learnable_skip":
            t.uniform_(0.9, 1.1)
        elif cls in norms and leaf == "weight":
            t.uniform_(0.9, 1.1)
        elif leaf == "bias" or cls in norms:
            t.uniform_(-0.05, 0.05)
        else:
            s = 1.0 / math.sqrt(t.shape[-1])
            t.uniform_(-s, s)


# ---------------------------------------------------------------- comparison

class Check:
    """One named check: accumulates the worst difference and the first failures."""

    def __init__(self, label, atol, rtol):
        self.label, self.atol, self.rtol = label, atol, rtol
        self.err = 0.0
        self.fails: list[str] = []

    def cmp(self, what, a, b):
        if a.numel():
            # where(a == b, 0, |a - b|) so two equal -inf count as agreeing
            # rather than as the nan their difference would be.
            d = (a.to(torch.float64) - b.to(torch.float64)).abs()
            self.err = max(self.err, float(torch.where(a == b, 0.0, d).max()))
        try:
            torch.testing.assert_close(a, b, atol=self.atol, rtol=self.rtol)
        except AssertionError as exc:
            self.note(f"{what}: {str(exc).strip().splitlines()[0]}")

    def note(self, msg):
        if len(self.fails) < 6:
            self.fails.append(msg)

    def report(self, extra="") -> bool:
        ok = not self.fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {self.label:<38} "
              f"max abs diff {self.err:.2e}{extra}")
        for f in self.fails:
            print(f"           {f}")
        return ok


def audit_identity(state, returned, chk, masked: bool) -> int:
    """With a mask, exactly the 4-D leaves must come back as the same object."""
    same = 0
    for o, n in zip(leaves(state), leaves(returned), strict=True):
        want = masked and o.dim() == BIG_DIM
        if (n is o) != want:
            chk.note(f"leaf {tuple(o.shape)} came back as "
                     f"{'the same' if n is o else 'a new'} object, wanted "
                     f"{'the same' if want else 'a new'} one")
        same += n is o
    return same


# --------------------------------------------------------------------- cases
# (name, builder, input width, steps). Mamba2's head sizes are overridden to
# small and unequal values, so a reduction over the wrong axis fails on shape
# instead of quietly returning something the right shape.

def _model(p):
    im = p["inference_model"]
    return im.MultiTowerModel(im.ModelConfig(hidden_size=64, proj_size=128,
                                             tower_depth=2, num_heads=4,
                                             num_features=7))


CASES = [
    ("xlstm", lambda p: p["xlstm"].XLSTM(64, mlstm_num_heads=4, slstm_num_heads=2), 64, 12),
    ("mamba2", lambda p: p["mamba2"].Mamba2(64, head_size=16, bc_head_size=24), 64, 12),
    ("retnet", lambda p: p["retnet"].RetNet(64, num_heads=4), 64, 12),
    ("hawk", lambda p: p["hawk"].Hawk(64), 64, 12),
    ("model", _model, 7, 6),
]


def make_pair(build, dev, seed):
    torch.manual_seed(seed)
    old = build(OLD).to(dev).eval()
    randomize_(old)
    new = build(NEW).to(dev).eval()
    new.load_state_dict(old.state_dict(), strict=True)  # proves the key sets match
    return old, new


def masked_case(name, build, width, steps, dev, atol, rtol, batch=6, inference=False):
    tag = " (inference_mode)" if inference else ""
    chk = Check(f"{name}: masked, {steps} steps{tag}", atol, rtol)
    old, new = make_pair(build, dev, 1234)
    s_old, s_new = old.init_state(batch, dev), new.init_state(batch, dev)
    identity = 0
    for step in range(steps):
        x = torch.randn(batch, width, device=dev)
        mask = random_mask(batch, dev)
        before = clone_tree(s_new)

        out_old, ret_old = old(x, s_old)
        s_old = blend_reference(s_old, ret_old, mask)

        with torch.inference_mode() if inference else contextlib.nullcontext():
            out_new, ret_new = new(x, s_new, mask=mask)
            identity = audit_identity(s_new, ret_new, chk, masked=True)
            blend_client(s_new, ret_new, mask)

        chk.cmp(f"step {step} out", out_old[mask], out_new[mask])
        for i, (a, b) in enumerate(zip(leaves(s_old), leaves(s_new), strict=True)):
            chk.cmp(f"step {step} state leaf {i}", a, b)
        idle = ~mask
        for i, (b0, b1) in enumerate(zip(before, leaves(s_new), strict=True)):
            if not torch.equal(b0[idle], b1[idle]):
                chk.note(f"step {step} leaf {i}: a masked-off row moved")
    return chk.report(f", {identity} leaves by identity")


def none_case(name, build, width, steps, dev, atol, rtol, batch=6):
    chk = Check(f"{name}: mask=None, {steps} steps", atol, rtol)
    old, new = make_pair(build, dev, 4321)
    s_old, s_new = old.init_state(batch, dev), new.init_state(batch, dev)
    for step in range(steps):
        x = torch.randn(batch, width, device=dev)
        before = clone_tree(s_new)
        out_old, s_old = old(x, s_old)
        out_new, ret = new(x, s_new)
        audit_identity(s_new, ret, chk, masked=False)
        for i, (b0, b1) in enumerate(zip(before, leaves(s_new), strict=True)):
            if not torch.equal(b0, b1):
                chk.note(f"step {step} leaf {i}: mask=None wrote the state it was given")
        s_new = ret
        chk.cmp(f"step {step} out", out_old, out_new)
        for i, (a, b) in enumerate(zip(leaves(s_old), leaves(s_new), strict=True)):
            chk.cmp(f"step {step} state leaf {i}", a, b)
    return chk.report()


def step_fn(model):
    """A plain function for dynamo to trace, so the mask arrives as an argument."""
    def step(x, st, m):
        return model(x, st, mask=m)
    return step


def compile_case(dev, steps=3, batch=6):
    """The same masked steps, eager then compiled, to catch a compile-time error."""
    chk = Check(f"model: torch.compile vs eager, {steps} steps", 1e-3, 1e-3)
    _, build, width, _ = CASES[-1]
    torch.manual_seed(7)
    model = build(NEW).to(dev).eval()
    randomize_(model)
    state = model.init_state(batch, dev)
    snap = clone_tree(state)
    xs = [torch.randn(batch, width, device=dev) for _ in range(steps)]
    masks = [random_mask(batch, dev) for _ in range(steps)]
    step = step_fn(model)

    preds = []
    for x, m in zip(xs, masks):
        p, ns = step(x, state, m)
        blend_client(state, ns, m)
        preds.append(p.clone())
    eager_state = clone_tree(state)

    restore_tree(state, snap)
    compiled = torch.compile(step, fullgraph=False)
    identity = 0
    for i, (x, m) in enumerate(zip(xs, masks)):
        p, ns = compiled(x, state, m)
        identity = sum(n is o for o, n in zip(leaves(state), leaves(ns), strict=True))
        blend_client(state, ns, m)
        chk.cmp(f"step {i} preds", preds[i], p)
    for i, (a, b) in enumerate(zip(eager_state, leaves(state), strict=True)):
        chk.cmp(f"final state leaf {i}", a, b)
    # Losing identity through dynamo costs the speedup, not the answer: the
    # blend then re-folds a value that is already folded, which is a no-op.
    return chk.report(f", {identity} leaves by identity through dynamo")


# --------------------------------------------------------------------- bench

def time_steps(fn, steps, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps


def bench(dev, batch=39, hidden=2048, heads=8, steps=20):
    """Real sizes, one layer at a time. Each timing includes that path's own fold."""
    im_old, im_new = OLD["inference_model"], NEW["inference_model"]
    print(f"\nbench: B={batch}, hidden={hidden}, {steps} masked steps, ms per step; "
          f"GB/s assumes the step moved 2x the state")
    print(f"  {'layer':<8}{'state':>9}{'old eager':>11}{'new eager':>11}"
          f"{'compiled':>11}{'compiled GB/s':>15}")
    for name in ("XLSTM", "MAMBA2", "RETNET"):
        x = torch.randn(batch, hidden, device=dev)
        mask = random_mask(batch, dev)

        old = im_old.create_layer(getattr(im_old.LayerType, name), hidden, heads)
        old = old.to(dev).eval()
        randomize_(old)
        s_old = old.init_state(batch, dev)
        nbytes = state_bytes(s_old)

        # blend_client on the old model is the old client: nothing it returns is
        # an identity leaf, so the skip never fires and every leaf pays its
        # torch.where plus its copy_, which is what the old round cost.
        def old_step():
            _, ns = old(x, s_old)
            blend_client(s_old, ns, mask)

        t_old = time_steps(old_step, steps)
        del old, s_old
        torch.cuda.empty_cache()

        new = im_new.create_layer(getattr(im_new.LayerType, name), hidden, heads)
        new = new.to(dev).eval()
        randomize_(new)
        s_new = new.init_state(batch, dev)
        step = step_fn(new)
        compiled = torch.compile(step, fullgraph=False)

        def new_step():
            _, ns = step(x, s_new, mask)
            blend_client(s_new, ns, mask)

        def comp_step():
            _, ns = compiled(x, s_new, mask)
            blend_client(s_new, ns, mask)

        t_new = time_steps(new_step, steps)
        try:
            ms = time_steps(comp_step, steps)
            t_comp = f"{ms:11.3f}"
            gbs = f"{2 * nbytes / (ms / 1000) / 1e9:15.1f}"
        except Exception as exc:
            t_comp, gbs = f"{'failed':>11}", f"{type(exc).__name__:>15}"
        print(f"  {name.lower():<8}{nbytes / 1024**3:8.3f}G{t_old:11.3f}{t_new:11.3f}"
              f"{t_comp}{gbs}")
        del new, s_new, step, compiled
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="3b2e25c", help="git ref holding the old model/")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if there)")
    ap.add_argument("--bench", action="store_true", help="also time real sizes (CUDA)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        # The old einsums lower to cuBLAS and the new sums do not, so with TF32
        # on, the two would differ by 2**-11 for a reason that has nothing to do
        # with the rewrite. Measure that separately, against the real checkpoint.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    atol, rtol = (1e-4, 1e-5) if dev.type == "cuda" else (1e-5, 1e-5)

    with tempfile.TemporaryDirectory() as tmp:
        OLD.update(load_old(args.ref, Path(tmp)))
        NEW.update(load_new())
        print(f"torch {torch.__version__}, device {dev}, old package from "
              f"{args.ref}, atol {atol:g} rtol {rtol:g}")

        results: list[bool] = []

        def run(label, fn, *a, **kw):
            try:
                results.append(fn(*a, **kw))
            except Exception:
                print(f"  [FAIL] {label} raised")
                traceback.print_exc()
                results.append(False)

        for name, build, width, steps in CASES:
            run(f"{name} masked", masked_case, name, build, width, steps, dev, atol, rtol)
            run(f"{name} mask=None", none_case, name, build, width, 4, dev, atol, rtol)

        name, build, width, steps = CASES[-1]
        run("model inference_mode", masked_case, name, build, width, steps,
            dev, atol, rtol, inference=True)

        if dev.type == "cuda":
            run("model torch.compile", compile_case, dev)
            if args.bench:
                bench(dev)
        else:
            print("  [skip] torch.compile and --bench need CUDA")

    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} checks passed - {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
