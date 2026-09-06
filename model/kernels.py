"""Fused Triton steps for the three big recurrent states.

Every round each of the three towers does the same three things to a
(B, n, h, k) fp32 state: an elementwise update (decay the old state, add a
rank-1 outer product), a contraction of the *new* state against a query vector,
and a masked write-back that has to leave idle rows bit-identical. Inductor
cannot fold those into one kernel - the new state has two consumers, so it is
materialized and read back - and it cannot skip idle rows, so all B rows cross
HBM every round even when eight of thirty-nine are active: on the A6000 the
mLSTM cell alone costs four passes over its state per block, 23 ms of a 42 ms
round.

Each kernel here reads a state element once and writes it once, with the
update, the contraction and the masked write done in registers in between, so
the new state is never materialized; and an inactive row is neither loaded nor
stored - the whole body sits under `if active` - so the cost follows the number
of active rows rather than the batch.

Which axis is contracted differs, and the tile shape follows from it. mLSTM and
RetNet contract dim 2 (h, the axis their k/q vectors index) and carry dim 3 (k,
the contiguous one, indexed by v) through to the output, so a program owns one
(row, head, k-tile) and walks h; the state loads are coalesced along k and the
reduction runs down the tile's columns. Mamba2 contracts dim 3 (c, the
contiguous one, indexed by b and c) and carries dim 2 (h, indexed by x), so its
tile is transposed - a program owns one (row, head, h-tile) and holds the whole
of c, closing the reduction inside the program.

Semantics, identical on the Triton and the reference path: the state is updated
IN PLACE and is not returned, rows with mask False ending bit-identical to how
they started and rows with mask True holding the new value up to floating-point
reassociation; the returned output holds zeros, never uninitialized memory, on
idle rows; mask None means every row is active; and nothing is data-dependent on
the host - no .item(), no nonzero, no boolean indexing on the mask, no shape
that depends on its values.

CELL_KERNELS=torch forces the reference path; the wrappers also fall back to it
when triton is missing or the state is not on CUDA, so importing this module is
safe anywhere.
"""

from __future__ import annotations

import os

import torch

try:  # absent on the dev box, and on CPU-only installs
    import triton
    import triton.language as tl

    AVAILABLE = True
except Exception:  # pragma: no cover - depends on the install, not on the code
    AVAILABLE = False

ENABLED = (
    AVAILABLE and os.environ.get("CELL_KERNELS", "triton").strip().lower() == "triton"
)

# The launch config, fixed rather than autotuned: autotune re-launches with
# several configs to pick one, which does not survive CUDA-graph capture.
# All three tiles are 32 x 128 fp32, 16 KB, which leaves room for the updated
# value held live beside the loaded one. These five constants are the whole
# tuning surface; sweep them here if ncu shows spills or a stall on shared
# memory. NUM_STAGES=2 keeps one tile prefetch in flight: with a 16 KB tile,
# three stages would put 32 KB of staging buffer per program against the 100 KB
# an sm_86 SM has, and this kernel wants many resident programs more than it
# wants a deeper pipeline. Try 3 before anything else if it lands short of
# 768 GB/s.
TILE_H = 32
TILE_K = 128
TILE_HC = 32
TILE_C = 128
NUM_WARPS = 4
NUM_STAGES = 2


def _tile(limit: int, extent: int) -> int:
    """A power-of-two block size, at least 16, no larger than `limit`."""
    size = 16
    while size < extent and size < limit:
        size *= 2
    return size


def _small(t: torch.Tensor, shape: tuple, name: str, device) -> torch.Tensor:
    """Check one read-only operand and hand back a contiguous view of it.

    Only the state may not be copied - it is mutated in place - so these are
    free to be materialized, and .contiguous() is a no-op on every one of them
    as the cells call it. fp32 is asserted rather than cast: CastLinear hands
    every module boundary back a float32 tensor even under --linear-dtype bf16,
    so a non-fp32 operand here means something upstream changed, and silently
    casting would hide it behind a full-size copy of v.
    """
    assert t.dtype == torch.float32, f"{name} must be fp32, got {t.dtype}"
    assert tuple(t.shape) == shape, f"{name} must be {shape}, got {tuple(t.shape)}"
    assert t.device == device, f"{name} must be on {device}, it is on {t.device}"
    return t.contiguous()


def _mask_i32(mask: torch.Tensor | None, batch: int, device) -> torch.Tensor:
    """The mask as int32: triton bool pointers are a known trouble spot.

    .contiguous() after the cast is load-bearing and not a no-op: .to(dtype)
    preserves the strides of a dense non-contiguous input, so a stepped view of
    a mask would otherwise be read by the kernel as if it were packed.
    """
    if mask is None:
        return torch.ones(batch, dtype=torch.int32, device=device)
    assert mask.dtype == torch.bool, f"mask must be bool, got {mask.dtype}"
    assert mask.shape == (batch,), f"mask must be ({batch},), got {tuple(mask.shape)}"
    assert mask.device == device, f"mask must be on {device}, it is on {mask.device}"
    return mask.to(torch.int32).contiguous()


def _state_ok(state: torch.Tensor, name: str) -> None:
    # Asserted, never fixed up: the state is mutated in place and handed back by
    # identity, so a .contiguous() copy here would drop the update on the floor.
    assert state.dtype == torch.float32, f"{name} must be fp32, got {state.dtype}"
    assert state.is_contiguous(), f"{name} must be contiguous, it is written in place"
    assert state.dim() == 4, f"{name} must be 4-D, got {tuple(state.shape)}"


def _write_back(state, new, out, mask, batch):
    """Masked in-place write of `new` into `state`; zeros in `out`'s idle rows.

    view(batch, ...) and not view(-1, ...): the -1 form takes a mask of the wrong
    length and broadcasts it, so a length-1 mask would advance every row in
    silence. torch.where selects rather than computes - a nan on an idle row
    cannot leak into a kept one - and builds the blend before copy_ writes it,
    which makes `state` safe as both a source and the destination.
    """
    if mask is None:
        state.copy_(new)
        return out
    state.copy_(torch.where(mask.view(batch, 1, 1, 1), new, state))
    return torch.where(mask.view(batch, 1, 1), out, out.new_zeros(()))


# --------------------------------------------------------------- torch path


def mlstm_step_reference(cell, q, k, v, i_gate, f_gate, mask):
    """cell <- f*cell + i*(k outer v) on active rows; return sum_h q*cell_new."""
    batch = cell.shape[0]
    cell_new = (
        f_gate[:, :, None, None] * cell
        + i_gate[:, :, None, None] * k[:, :, :, None] * v[:, :, None, :]
    )
    numerator = (q[:, :, :, None] * cell_new).sum(dim=2)
    return _write_back(cell, cell_new, numerator, mask, batch)


def retnet_step_reference(state, k_rope, v, q_scaled, decay, mask):
    """state <- decay*state + (k outer v); return sum_h q_scaled*state_new."""
    batch = state.shape[0]
    outer = k_rope[:, :, :, None] * v[:, :, None, :]
    state_new = state * decay[None, :, None, None] + outer
    out = (q_scaled[:, :, :, None] * state_new).sum(dim=2)
    return _write_back(state, state_new, out, mask, batch)


def mamba2_step_reference(ssm, x, b, c, dt, decay, mask):
    """ssm <- decay*ssm + dt*(b outer x); return sum_c c*ssm_new."""
    batch = ssm.shape[0]
    ssm_new = (
        decay[:, :, None, None] * ssm
        + dt[:, :, None, None] * b[:, None, None, :] * x[:, :, :, None]
    )
    y = (c[:, None, None, :] * ssm_new).sum(dim=-1)
    return _write_back(ssm, ssm_new, y, mask, batch)


# ------------------------------------------------------------- triton path

if AVAILABLE:

    @triton.jit
    def _mlstm_step_kernel(
        cell_ptr, q_ptr, k_ptr, v_ptr, i_ptr, f_ptr, mask_ptr, num_ptr,
        N, H, K,
        stride_b, stride_n, stride_h, stride_k,
        BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // N
        head = pid % N
        offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        in_k = offs_k < K
        # rn addresses the (b, n) plane of the small contiguous operands and stays
        # int32: B*n*K is a few hundred thousand. row is promoted for the state
        # base alone, because row*stride_b is the one product that can grow - it
        # is already 8e7 at B=39 - and the in-tile offsets then ride on a pointer.
        rn = row * N + head
        row64 = row.to(tl.int64)
        out_ptrs = num_ptr + rn * K + offs_k

        # The whole body, loads included, hangs off this branch: an idle row costs
        # no traffic at all, which is the half of the win fusion alone cannot buy.
        # Its output is still written - zeros, not whatever the allocator held.
        # If a Triton release ever refuses the loop nested in this scf.if, the
        # equivalent without a branch is to keep acc at zero and clamp the trip
        # count: `for h0 in range(0, tl.where(active, H, 0), BLOCK_H)`, then store
        # acc unconditionally. Same traffic, one store path, no scf.if.
        active = tl.load(mask_ptr + row) != 0
        if active:
            i_g = tl.load(i_ptr + rn)
            f_g = tl.load(f_ptr + rn)
            v_row = tl.load(v_ptr + rn * K + offs_k, mask=in_k, other=0.0)
            base = cell_ptr + row64 * stride_b + head * stride_n
            acc = tl.zeros([BLOCK_K], dtype=tl.float32)
            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                in_h = offs_h < H
                kk = tl.load(k_ptr + rn * H + offs_h, mask=in_h, other=0.0)
                qq = tl.load(q_ptr + rn * H + offs_h, mask=in_h, other=0.0)
                ptrs = base + offs_h[:, None] * stride_h + offs_k[None, :] * stride_k
                tile = in_h[:, None] & in_k[None, :]
                old = tl.load(ptrs, mask=tile, other=0.0)
                # (i*k)*v, associated as the torch path writes it. Rows and columns
                # past the extent hold 0 here and contribute 0 to acc; the store
                # masks them off, so a partial tile costs nothing and writes nothing.
                new = f_g * old + (i_g * kk[:, None]) * v_row[None, :]
                acc += tl.sum(qq[:, None] * new, axis=0)
                tl.store(ptrs, new, mask=tile)
            tl.store(out_ptrs, acc, mask=in_k)
        else:
            tl.store(out_ptrs, tl.zeros([BLOCK_K], dtype=tl.float32), mask=in_k)

    @triton.jit
    def _retnet_step_kernel(
        st_ptr, k_ptr, v_ptr, q_ptr, decay_ptr, mask_ptr, out_ptr,
        N, H, K,
        stride_b, stride_n, stride_h, stride_k,
        BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // N
        head = pid % N
        offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        in_k = offs_k < K
        rn = row * N + head
        row64 = row.to(tl.int64)
        out_ptrs = out_ptr + rn * K + offs_k

        active = tl.load(mask_ptr + row) != 0
        if active:
            # decay is a buffer of shape (n,) - one value per head, shared by every
            # row - so it is indexed by head alone, with no row term. That and the
            # missing input gate are the only places this differs from the mLSTM.
            dec = tl.load(decay_ptr + head)
            v_row = tl.load(v_ptr + rn * K + offs_k, mask=in_k, other=0.0)
            base = st_ptr + row64 * stride_b + head * stride_n
            acc = tl.zeros([BLOCK_K], dtype=tl.float32)
            for h0 in range(0, H, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                in_h = offs_h < H
                kk = tl.load(k_ptr + rn * H + offs_h, mask=in_h, other=0.0)
                qq = tl.load(q_ptr + rn * H + offs_h, mask=in_h, other=0.0)
                ptrs = base + offs_h[:, None] * stride_h + offs_k[None, :] * stride_k
                tile = in_h[:, None] & in_k[None, :]
                old = tl.load(ptrs, mask=tile, other=0.0)
                new = old * dec + kk[:, None] * v_row[None, :]
                # Sum over h, dim 2 - the axis k_rope indexes. Both trailing axes
                # are head_size, so contracting dim 3 would have the right shape
                # and the wrong value; this reduces the tile's rows, not its
                # columns, which is why h is the row axis of the tile.
                acc += tl.sum(qq[:, None] * new, axis=0)
                tl.store(ptrs, new, mask=tile)
            tl.store(out_ptrs, acc, mask=in_k)
        else:
            tl.store(out_ptrs, tl.zeros([BLOCK_K], dtype=tl.float32), mask=in_k)

    @triton.jit
    def _mamba2_step_kernel(
        ssm_ptr, x_ptr, b_ptr, c_ptr, dt_ptr, decay_ptr, mask_ptr, y_ptr,
        N, H, C,
        stride_b, stride_n, stride_h, stride_c,
        BLOCK_H: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // N
        head = pid % N
        offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        in_h = offs_h < H
        rn = row * N + head
        row64 = row.to(tl.int64)
        y_ptrs = y_ptr + rn * H + offs_h

        # This one contracts over c, the contiguous axis, so the reduction runs
        # along the tile's rows instead of down its columns and the program owns a
        # strip of h rather than of the output axis. The whole of c fits in one
        # tile at the real size (128), so the loop closes in a single trip; it is
        # written as a loop anyway so that a larger c still reduces correctly
        # rather than demanding a tile nobody can hold.
        active = tl.load(mask_ptr + row) != 0
        if active:
            # dt and decay are both (B, n) - indexed by row AND head, unlike
            # RetNet's decay - and b and c are (B, c), indexed by row alone.
            dt_v = tl.load(dt_ptr + rn)
            dec = tl.load(decay_ptr + rn)
            x_row = tl.load(x_ptr + rn * H + offs_h, mask=in_h, other=0.0)
            base = ssm_ptr + row64 * stride_b + head * stride_n
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for c0 in range(0, C, BLOCK_C):
                offs_c = c0 + tl.arange(0, BLOCK_C)
                in_c = offs_c < C
                bv = tl.load(b_ptr + row * C + offs_c, mask=in_c, other=0.0)
                cv = tl.load(c_ptr + row * C + offs_c, mask=in_c, other=0.0)
                ptrs = base + offs_h[:, None] * stride_h + offs_c[None, :] * stride_c
                tile = in_h[:, None] & in_c[None, :]
                old = tl.load(ptrs, mask=tile, other=0.0)
                new = dec * old + (dt_v * bv)[None, :] * x_row[:, None]
                acc += tl.sum(cv[None, :] * new, axis=1)
                tl.store(ptrs, new, mask=tile)
            tl.store(y_ptrs, acc, mask=in_h)
        else:
            tl.store(y_ptrs, tl.zeros([BLOCK_H], dtype=tl.float32), mask=in_h)


# ---------------------------------------------------------------- wrappers


def _launch_hk(kernel, state, inputs, mask):
    """Launch one of the two h-contracting kernels.

    They differ only in the inputs between the state and the mask, so the state
    check, the tiling, the grid and the (B, n, k) output are shared. Grid and
    tiles are Python ints and the constexprs go in by keyword, which is what
    torch.compile needs to trace this as a user-defined Triton kernel.
    """
    batch, heads, size_h, size_k = (int(s) for s in state.shape)
    out = torch.empty((batch, heads, size_k), dtype=torch.float32, device=state.device)
    block_h, block_k = _tile(TILE_H, size_h), _tile(TILE_K, size_k)
    grid = (batch * heads, (size_k + block_k - 1) // block_k)
    kernel[grid](
        state, *inputs, _mask_i32(mask, batch, state.device), out,
        heads, size_h, size_k,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        BLOCK_H=block_h, BLOCK_K=block_k,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return out


def mlstm_step(cell, q, k, v, i_gate, f_gate, mask):
    """Fused mLSTM cell step. See the module docstring for the contract."""
    if not (ENABLED and cell.is_cuda):
        return mlstm_step_reference(cell, q, k, v, i_gate, f_gate, mask)
    _state_ok(cell, "cell")
    batch, heads, size_h, size_k = (int(s) for s in cell.shape)
    dev = cell.device
    prepared = (
        _small(q, (batch, heads, size_h), "q", dev),
        _small(k, (batch, heads, size_h), "k", dev),
        _small(v, (batch, heads, size_k), "v", dev),
        _small(i_gate, (batch, heads), "i_gate", dev),
        _small(f_gate, (batch, heads), "f_gate", dev),
    )
    return _launch_hk(_mlstm_step_kernel, cell, prepared, mask)


def retnet_step(state, k_rope, v, q_scaled, decay, mask):
    """Fused RetNet recurrent step. See the module docstring for the contract."""
    if not (ENABLED and state.is_cuda):
        return retnet_step_reference(state, k_rope, v, q_scaled, decay, mask)
    _state_ok(state, "recurrent_state")
    batch, heads, size_h, size_k = (int(s) for s in state.shape)
    dev = state.device
    prepared = (
        _small(k_rope, (batch, heads, size_h), "k_rope", dev),
        _small(v, (batch, heads, size_k), "v", dev),
        _small(q_scaled, (batch, heads, size_h), "q_scaled", dev),
        _small(decay, (heads,), "decay", dev),
    )
    return _launch_hk(_retnet_step_kernel, state, prepared, mask)


def mamba2_step(ssm, x, b, c, dt, decay, mask):
    """Fused Mamba2 SSM step. See the module docstring for the contract."""
    if not (ENABLED and ssm.is_cuda):
        return mamba2_step_reference(ssm, x, b, c, dt, decay, mask)
    _state_ok(ssm, "ssm_state")
    batch, heads, size_h, size_c = (int(s) for s in ssm.shape)
    dev = ssm.device
    x = _small(x, (batch, heads, size_h), "x", dev)
    b = _small(b, (batch, size_c), "b", dev)
    c = _small(c, (batch, size_c), "c", dev)
    dt = _small(dt, (batch, heads), "dt", dev)
    decay = _small(decay, (batch, heads), "decay", dev)
    y = torch.empty((batch, heads, size_h), dtype=torch.float32, device=dev)
    block_h, block_c = _tile(TILE_HC, size_h), _tile(TILE_C, size_c)
    grid = (batch * heads, (size_h + block_h - 1) // block_h)
    _mamba2_step_kernel[grid](
        ssm, x, b, c, dt, decay, _mask_i32(mask, batch, dev), y,
        heads, size_h, size_c,
        ssm.stride(0), ssm.stride(1), ssm.stride(2), ssm.stride(3),
        BLOCK_H=block_h, BLOCK_C=block_c,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return y
