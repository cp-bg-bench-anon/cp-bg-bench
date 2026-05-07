"""
Minimal vendor of SubCellPortable/vit_model.py (CellProfiling/SubCellPortable, Apache-2.0).
Source: https://github.com/CellProfiling/SubCellPortable
Kept as close to the original as possible; only unused classifier helpers are omitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.models.vit.modeling_vit import (
    BaseModelOutputWithPooling,
    ViTAttention,
    ViTEmbeddings,
    ViTIntermediate,
    ViTOutput,
    ViTPatchEmbeddings,
    ViTPooler,
    ViTPreTrainedModel,
)

logger = logging.getLogger(__name__)


@dataclass
class ViTPoolModelOutput:
    attentions: tuple[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pool_op: torch.FloatTensor = None
    pool_attn: torch.FloatTensor = None
    probabilities: torch.FloatTensor = None


class GatedAttentionPooler(nn.Module):
    def __init__(self, dim: int, int_dim: int = 512, num_heads: int = 1, out_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.attention_v = nn.Sequential(nn.Linear(dim, int_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, int_dim), nn.GELU())
        self.attention = nn.Linear(int_dim, num_heads)
        self.softmax = nn.Softmax(dim=-1)
        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)
        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)
        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)
        x = self.out_proj(x)
        return x, attn


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig, sdpa_attn: bool = False) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor]:
        # output_attentions ignored: newer transformers ViTAttention doesn't return attn weights
        attention_output = self.attention(self.layernorm_before(hidden_states), head_mask)
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,)


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(ViTLayer(config, sdpa_attn=(i < config.num_hidden_layers - 1)))
        self.layer = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__, hidden_states, layer_head_mask, output_attentions
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTInferenceModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTPoolClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        vit_model_config = config["vit_model"].copy()
        vit_model_config["attn_implementation"] = "eager"
        self.vit_config = ViTConfig(**vit_model_config)
        self.encoder = ViTInferenceModel(self.vit_config, add_pooling_layer=False)
        pool_config = config.get("pool_model")
        self.pool_model = GatedAttentionPooler(**pool_config) if pool_config else None
        self.out_dim = self.pool_model.out_dim if self.pool_model else self.vit_config.hidden_size
        self.num_classes = config["num_classes"]
        # classifiers not loaded by default; forward handles missing gracefully
        self.classifiers: nn.ModuleList = nn.ModuleList([])

    def load_encoder_weights(self, encoder_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder_ckpt = {k[len("encoder.") :]: v for k, v in checkpoint.items() if "encoder." in k}
        status = self.encoder.load_state_dict(encoder_ckpt)
        logger.info(f"SubCell encoder status: {status}")

        pool_ckpt = {k.replace("pool_model.", ""): v for k, v in checkpoint.items() if "pool_model." in k}
        pool_ckpt = {k.replace("1.", "0."): v for k, v in pool_ckpt.items()}
        if pool_ckpt and self.pool_model is not None:
            status = self.pool_model.load_state_dict(pool_ckpt)
            logger.info(f"SubCell pool model status: {status}")

    def forward(self, x: torch.Tensor) -> ViTPoolModelOutput:
        outputs = self.encoder(x, output_attentions=False, interpolate_pos_encoding=True)

        if self.pool_model is not None:
            pool_op, pool_attn = self.pool_model(outputs.last_hidden_state)
        else:
            pool_op = torch.mean(outputs.last_hidden_state, dim=1)
            pool_attn = None

        probs = torch.zeros(pool_op.shape[0], self.num_classes, device=pool_op.device)

        return ViTPoolModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            attentions=None,
            pool_op=pool_op,
            pool_attn=pool_attn,
            probabilities=probs,
        )
