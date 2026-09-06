from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from .mamba2 import Mamba2
from .retnet import RetNet
from .hawk import Hawk
from .xlstm import XLSTM
from .modules import SimpleMLP


class LayerType(Enum):
    MAMBA2 = "mamba2"
    RETNET = "retnet"
    HAWK = "hawk"
    XLSTM = "xlstm"


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    hidden_size: int
    proj_size: int
    tower_depth: int
    num_heads: int
    num_features: int


def create_layer(layer_type: LayerType, hidden_size: int, num_heads: int = 8):
    """Factory function to create baseline models

    Args:
        layer_type: Type of layer to create
        hidden_size: Hidden dimension
        num_heads: Number of attention heads (for models that use it)
    """
    match layer_type:
        case LayerType.MAMBA2:
            return Mamba2(hidden_size=hidden_size)
        case LayerType.RETNET:
            return RetNet(hidden_size=hidden_size, num_heads=num_heads)
        case LayerType.HAWK:
            return Hawk(hidden_size=hidden_size)
        case LayerType.XLSTM:
            return XLSTM(
                hidden_size=hidden_size,
                mlstm_num_heads=num_heads,
                slstm_num_heads=num_heads // 2,
            )
        case _:
            raise ValueError(f"Unknown layer type: {layer_type}.")


class Block(nn.Module):
    def __init__(self, layer_type: LayerType, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.layer = create_layer(layer_type, config.hidden_size, config.num_heads)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.mlp = SimpleMLP(config.hidden_size, config.proj_size)

    def forward(self, x: torch.Tensor, state, mask=None):
        skip = x
        x, new_state = self.layer(self.norm1(x), state, mask=mask)
        x = x + skip
        x = x + self.mlp(self.norm2(x))
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device):
        return self.layer.init_state(batch_size, device)


class Tower(nn.Module):
    def __init__(self, config: ModelConfig, layer_type):
        super().__init__()

        self.input_up = nn.Linear(config.num_features, config.proj_size)
        self.input_down = nn.Linear(config.proj_size, config.hidden_size)

        self.blocks = nn.ModuleList(
            [Block(layer_type, config) for _ in range(config.tower_depth)]
        )

    def forward(self, x: torch.Tensor, state, mask=None):
        new_state = []

        x = self.input_up(x)
        x = nn.functional.relu(x)
        x = self.input_down(x)

        for block, block_state in zip(self.blocks, state, strict=True):
            x, new_block_state = block(x, block_state, mask=mask)
            new_state.append(new_block_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device):
        return [
            block.init_state(batch_size, device) for block in self.blocks
        ]  # pyright: ignore[reportCallIssue]


class MultiTowerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        layer_types = [
            LayerType.XLSTM,
            LayerType.MAMBA2,
            LayerType.RETNET,
            LayerType.HAWK,
        ]

        self.towers = nn.ModuleList(
            [Tower(config, layer_type) for layer_type in layer_types]
        )
        self.output_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x: torch.Tensor, state, mask=None):
        """Run every tower over the whole batch and concatenate their outputs.

        `mask` is either None or a bool (B,) tensor, True for the rows whose
        recurrent state must advance. It is threaded down to the cells rather
        than applied out here because the three tensors that dominate the state
        - the xLSTM mLSTM cell, the Mamba2 ssm_state, the RetNet
        recurrent_state - can then be updated in place, inside the same kernel
        that computes them, instead of being written once and re-read twice by
        a separate blend. Those leaves come back as the SAME tensor objects
        that went in; every other leaf is a fresh tensor, exactly as before,
        and the caller folds those itself.

        With mask=None nothing is written in place and the returned state is
        all fresh tensors, which is the shape training and any non-masked
        caller expect.
        """
        results = [
            tower(x, tower_state, mask=mask)
            for tower_state, tower in zip(state, self.towers, strict=True)
        ]
        xs, new_state = zip(*results, strict=True)

        xs = [self.output_proj(x) for x in xs]
        return torch.concat(xs, dim=1), list(new_state)

    def init_state(self, batch_size, device):
        return [tower.init_state(batch_size, device) for tower in self.towers]
