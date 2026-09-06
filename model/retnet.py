import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .modules import get_model_device


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x_rot


class RetNet(nn.Module):
    decay: torch.Tensor
    angle: torch.Tensor

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.scaling = self.head_size**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm = nn.RMSNorm(self.head_size, eps=1e-6, elementwise_affine=False)

        self.register_buffer("decay", torch.empty(num_heads))
        self.register_buffer("angle", torch.empty(self.head_size))

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        batch_size, hidden_size = x.shape
        assert hidden_size == self.hidden_size

        seq_offsets, scales, recurrent_state = state
        assert seq_offsets.shape == (batch_size,)
        assert scales.shape == (batch_size, self.num_heads)
        assert recurrent_state.shape == (
            batch_size,
            self.num_heads,
            self.head_size,
            self.head_size,
        )
        if mask is not None:
            # Shape only, like the asserts above, and checked for the same reason.
            # The torch write-back below broadcasts the mask against the state, so a
            # mask of the wrong length does not raise - a length-1 mask broadcasts
            # over the batch and advances every row, corrupting idle symbols in
            # silence. The Triton path indexes the mask by row instead, where a short
            # mask reads off the end of it. `is not None` tests the object, never the
            # values, so this adds no host sync and nothing data-dependent.
            assert mask.shape == (batch_size,)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k = k * self.scaling

        q_heads = q.view(batch_size, self.num_heads, self.head_size)
        k_heads = k.view(batch_size, self.num_heads, self.head_size)
        v_heads = v.view(batch_size, self.num_heads, self.head_size)

        # Rope
        sin = torch.sin(seq_offsets[:, None, None] * self.angle[None, None, :])
        cos = torch.cos(seq_offsets[:, None, None] * self.angle[None, None, :])

        q_rope = q_heads * cos + rotate_every_two(q_heads) * sin
        k_rope = k_heads * cos + rotate_every_two(k_heads) * sin

        # State scaling. The scale is one number per (row, head), constant over both
        # head-size axes, so it is built at (B, n, 1) - the rank its only consumer,
        # q_rope, needs. It used to be (B, n, 1, 1) to broadcast against the state;
        # carrying that rank here and squeezing it back would leave a shape-dependent
        # op between the definition and the use, and getting it wrong is not loud:
        # (B, n, 1, 1) * (B, n, h) broadcasts to (B, n, 1, h) rather than raising.
        # This is computed before the state update rather than after it because the
        # scaled q is an input to the fused kernel below; it reads nothing the update
        # writes, so the order between them is free.
        new_scales = scales * self.decay + 1.0
        scale_factor = (1.0 / new_scales.sqrt())[:, :, None]

        # The contraction is sum_h q_rope[b, n, h] * scale_factor[b, n] *
        # new_recurrent_state[b, n, h, k], and scale_factor is constant over (h, k), so
        # the scale rides on q instead of on the state. Scaling the state built a second
        # (B, n, h, h) tensor - a full extra read and write of the largest thing in this
        # layer that no compiler can remove - where scaling q costs (B, n, h).
        q_scaled = q_rope * scale_factor

        # State update and out. Both paths keep the update, the contraction and the
        # masked write-back together inside their branch: new_recurrent_state is the
        # largest tensor in this layer, and computing it on the kernel path would give
        # back exactly the passes the kernel exists to remove.
        if mask is not None and kernels.ENABLED and recurrent_state.is_cuda:
            # One Triton kernel: each element of the state is read once and written
            # once, with the decay-and-outer-product update, the contraction over h
            # and the masked write done in registers between the load and the store,
            # so new_recurrent_state is never materialized. A row whose mask is false
            # is neither loaded nor stored, so the cost follows the active row count
            # rather than the batch. recurrent_state is mutated in place and returned
            # as the same object, exactly as the torch path below, which is what tells
            # the client's _blend_state to skip this leaf. `out` for an idle row comes
            # back as zeros rather than as uninitialized memory; the client discards
            # those rows either way.
            out = kernels.retnet_step(
                recurrent_state, k_rope, v_heads, q_scaled, self.decay, mask
            )
            assert out.shape == (batch_size, self.num_heads, self.head_size)
            state_out = recurrent_state
        else:
            kv_outer_prod = k_rope[:, :, :, None] * v_heads[:, :, None, :]
            new_recurrent_state = (
                recurrent_state * self.decay[None, :, None, None] + kv_outer_prod
            )

            # Same contraction as torch.einsum("bnh,bnhk->bnk", ...), written as a
            # broadcast multiply plus a sum over h (dim 2 - the axis that came from
            # k_rope, which is the one the einsum contracted; both trailing axes are
            # head_size, so a sum over dim 3 would have the right shape and the wrong
            # value). einsum lowers to an extern bmm, which forces the state to be
            # materialized and read back from HBM. The sum form stays inside inductor,
            # so the state update above, this contraction and the masked write-back
            # below can fuse into one or two passes over the recurrent state.
            #
            # This also changes the arithmetic, and by more than reassociation: the
            # client sets matmul precision "high" and allow_tf32, so the bmm ran this
            # 256-long contraction in TF32 (10 mantissa bits) while a multiply plus a
            # reduction is not a matmul and neither flag reaches it - it runs in full
            # fp32. Expect the RetNet tower's error against tiny.parquet's targets to
            # move on the order of TF32's 2**-11, most likely downwards. Measure it; do
            # not assume. The Triton path above accumulates in fp32 too, so it sits on
            # this side of that change, not the bmm's.
            out = (q_scaled[:, :, :, None] * new_recurrent_state).sum(dim=2)

            if mask is not None:
                # Every read of the old recurrent_state (the state update above) and of
                # new_recurrent_state (the contraction) is done by this point, so the
                # masked update can land in place: rows with mask False keep their old
                # value bit-for-bit, rows with mask True take new_recurrent_state.
                # Returning the same tensor object tells the client's _blend_state to
                # skip this leaf, which drops another read of both states plus a full
                # write. The small seq_offsets and scales leaves stay fresh and are
                # blended by the client.
                recurrent_state.copy_(
                    torch.where(
                        mask.view(batch_size, 1, 1, 1),
                        new_recurrent_state,
                        recurrent_state,
                    )
                )
                state_out = recurrent_state
            else:
                state_out = new_recurrent_state

        out = self.norm(out).reshape(batch_size, self.hidden_size)
        out = F.silu(g) * out
        out = self.out_proj(out)

        return out, (seq_offsets + 1, new_scales, state_out)

    def init_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = get_model_device(self)
        return (
            torch.zeros(batch_size, dtype=torch.int32, device=device),
            torch.zeros(batch_size, self.num_heads, device=device),
            torch.zeros(
                batch_size,
                self.num_heads,
                self.head_size,
                self.head_size,
                device=device,
            ),
        )
