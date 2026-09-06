import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .modules import CausalConv1d, get_model_device


class Mamba2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int | None = None,
        head_size: int = 64,
        bc_head_size: int = 128,
        conv_kernel_size: int = 4,
    ):
        super().__init__()

        self.head_size = head_size
        self.bc_head_size = bc_head_size
        if inner_size is None:
            inner_size = 2 * hidden_size
        assert inner_size % head_size == 0
        self.inner_size = inner_size
        self.num_heads = inner_size // head_size

        # Projections
        self.input_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.z_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, bc_head_size, bias=False)
        self.c_proj = nn.Linear(hidden_size, bc_head_size, bias=False)
        self.dt_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        # Convs
        self.input_conv = CausalConv1d(inner_size, conv_kernel_size)
        self.b_conv = CausalConv1d(bc_head_size, conv_kernel_size)
        self.c_conv = CausalConv1d(bc_head_size, conv_kernel_size)

        # Other parameters
        self.a = nn.Parameter(-torch.empty(self.num_heads).uniform_(1, 16))
        self.d = nn.Parameter(torch.ones(self.num_heads))

        # Output
        self.norm = nn.RMSNorm(inner_size, eps=1e-5)
        self.out_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        # State for the convolutional layers
        conv_states = [
            conv.init_state(batch_size, device)
            for conv in [self.input_conv, self.b_conv, self.c_conv]
        ]
        # State of the SSM block
        ssm_state = torch.zeros(
            batch_size, self.num_heads, self.head_size, self.bc_head_size, device=device
        )
        return conv_states + [ssm_state]

    def forward(
        self,
        t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ):
        batch_size = t.shape[0]
        if mask is not None:
            # Shape only, and no host sync - `is not None` tests the object, never
            # the values. Both write-back paths need the length named rather than
            # inferred: the torch one broadcasts the mask against the state, where a
            # length-1 mask would advance every row in silence, and the Triton one
            # indexes it per row, where a short mask reads off the end of it.
            assert mask.shape == (batch_size,)

        x = self.input_proj(t)
        z = self.z_proj(t)
        b = self.b_proj(t)
        c = self.c_proj(t)
        dt = self.dt_proj(t)

        x_conv_state, b_conv_state, c_conv_state, ssm_state = state
        x, x_conv_state = self.input_conv(x, x_conv_state)
        b, b_conv_state = self.b_conv(b, b_conv_state)
        c, c_conv_state = self.c_conv(c, c_conv_state)
        x = F.silu(x)
        b = F.silu(b)
        c = F.silu(c)

        x = x.view(batch_size, self.num_heads, self.head_size)
        dt = F.softplus(dt)

        # SSM computation, this implements the discretized state space model.
        # new_state computation: h[t] = exp(A*dt) * h[t-1] + dt * B * x[t]
        # [batch_size, num_heads]
        decay = torch.exp(self.a[None] * dt)

        # State update, output contraction and masked write-back, kept together inside
        # whichever branch runs. ssm_new is the largest tensor in this layer, so
        # computing it above the branch to share it would give back exactly the passes
        # the fused kernel exists to remove.
        #
        # output computation: y[t] = C @ h[t] + D * x[t]
        # The accumulation in the product of C and h[t] is on the bc_head_size
        # dimension.
        if mask is not None and kernels.ENABLED and ssm_state.is_cuda:
            # One Triton kernel: each element of the state is read once and written
            # once, with the discretized update, the contraction over bc_head_size and
            # the masked write done in registers between the load and the store, so
            # ssm_new is never materialized. A row whose mask is false is neither
            # loaded nor stored, so the cost follows the number of active rows rather
            # than the batch. ssm_state is mutated in place and returned as the same
            # object, exactly as the torch path below, which is what makes the client's
            # _blend_state skip this leaf. state_contrib for an idle row comes back as
            # zeros rather than as uninitialized memory; the client discards those rows
            # either way.
            state_contrib = kernels.mamba2_step(ssm_state, x, b, c, dt, decay, mask)
            assert state_contrib.shape == (batch_size, self.num_heads, self.head_size)
            ssm_out = ssm_state
        else:
            # Broadcasting everything to the right shapes:
            #  dt is [batch_size, num_heads]
            #  b  is [batch_size, bc_head_size]
            #  x  is [batch_size, head_size]
            # The new contribution (and ssm_state) is
            # [batch_size, num_heads, head_size, bc_head_size]
            new_state_contrib = (
                dt[:, :, None, None] * b[:, None, None] * x[:, :, :, None]
            )
            # Keep the old ssm_state addressable: the masked write-back below needs it,
            # and it is the tensor we update in place.
            ssm_new = decay[:, :, None, None] * ssm_state + new_state_contrib

            # This is torch.einsum("bc,bnhc->bnh", c, ssm_new), written as a broadcast
            # multiply plus a reduction over the last (contiguous) dim so inductor
            # lowers it to a fused reduction instead of an extern bmm: the state update
            # above, this contraction and the write-back below then read the
            # (B, num_heads, head_size, bc_head_size) state once or twice instead of
            # three to six times.
            state_contrib = (c[:, None, None, :] * ssm_new).sum(dim=-1)

            # Masked write-back, after every read of the old ssm_state is done.
            # Updating the big state in place here lets the client skip blending this
            # leaf (it compares by identity), so the update, the contraction and the
            # write-back can fuse instead of each making its own pass over the state.
            # Rows with mask False keep their old value bit-for-bit. The small conv
            # states stay fresh tensors and are blended by the client as before.
            if mask is not None:
                ssm_state.copy_(
                    torch.where(mask.view(batch_size, 1, 1, 1), ssm_new, ssm_state)
                )
                ssm_out = ssm_state
            else:
                ssm_out = ssm_new

        # d has shape [num_heads], broadcasting it to the shape of x.
        y = state_contrib + self.d[None, :, None] * x

        # Combine heads
        y = y.view(batch_size, self.inner_size)
        # Gate, normalization and out
        y = y * F.silu(z)
        y = self.norm(y)
        output = self.out_proj(y)

        new_state = [x_conv_state, b_conv_state, c_conv_state, ssm_out]
        return output, new_state
