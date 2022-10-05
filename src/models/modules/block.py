# Copyright (c) 2022, Tri Dao.

from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.modules.mha import MHA
from src.models.modules.mlp import Mlp

try:
    from src.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None


class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout=0.,
                 fused_dropout_add_ln=False):
        super().__init__()
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.norm1 = norm_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.norm2 = norm_cls(dim)
            self.dropout2 = dropout_cls(resid_dropout)

        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_add_ln is not installed'
            assert isinstance(self.norm1, nn.LayerNorm) and isinstance(self.dropout1, nn.Dropout)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = LayerNorm(residual)
        """
        if self.prenorm:
            assert residual is not None
            mixer_out = self.mixer(hidden_states)
            if not self.fused_dropout_add_ln:
                residual = self.dropout1(mixer_out) + residual
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            else:
                hidden_states, residual = dropout_add_layer_norm(
                    mixer_out, residual, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps, prenorm=True
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if not self.fused_dropout_add_ln:
                    residual = self.dropout2(mlp_out) + residual
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                else:
                    hidden_states, residual = dropout_add_layer_norm(
                        mlp_out, residual, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps, prenorm=True
                    )
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(hidden_states)
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1((self.dropout1(mixer_out)
                                            + hidden_states).to(dtype=self.norm1.weight.dtype))
            else:
                hidden_states = dropout_add_layer_norm(
                    mixer_out, hidden_states, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps, prenorm=False
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2((self.drop_path(self.dropout2(mlp_out))
                                                + hidden_states).to(dtype=self.norm2.weight.dtype))
                else:
                    hidden_states = dropout_add_layer_norm(
                        mlp_out, hidden_states, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps, prenorm=False
                    )
            return hidden_states
