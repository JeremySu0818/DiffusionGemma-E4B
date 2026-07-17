from __future__ import annotations

from collections import UserDict

import torch
from torch import nn
from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaBlockDiffusionOutputWithPast,
    DiffusionGemmaDecoderModel,
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaGenerationMixin,
    DiffusionGemmaModelOutputWithPast,
    DiffusionGemmaMultimodalEmbedder,
    DiffusionGemmaPreTrainedModel,
    DiffusionGemmaSelfConditioning,
)
from transformers.models.gemma4.modeling_gemma4 import (
    ALL_ATTENTION_FUNCTIONS,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def _coerce_auto_config(config):
    if config is None or not isinstance(config, dict):
        return config
    model_type = config.get("model_type")
    if not model_type:
        return config
    return CONFIG_MAPPING[model_type](**config)


class MultimodalDiffusionGemmaEncoderModel(DiffusionGemmaPreTrainedModel):
    """DiffusionGemma encoder with Gemma4-compatible image and audio input merging."""

    accepts_loss_kwargs = False
    input_modalities = ("image", "text", "audio")

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        # Preserve Gemma 4 E4B's dense MLP, PLE and shared-KV prompt encoder.
        self.language_model = Gemma4TextModel(config=config.text_config)

        vision_config = _coerce_auto_config(getattr(config, "vision_config", None))
        audio_config = _coerce_auto_config(getattr(config, "audio_config", None))
        self.vision_tower = AutoModel.from_config(vision_config) if vision_config is not None else None
        self.audio_tower = AutoModel.from_config(audio_config) if audio_config is not None else None
        self.embed_vision = (
            DiffusionGemmaMultimodalEmbedder(vision_config, config.text_config)
            if vision_config is not None
            else None
        )
        self.embed_audio = (
            DiffusionGemmaMultimodalEmbedder(audio_config, config.text_config)
            if audio_config is not None
            else None
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError("Image features were requested, but the model has no vision tower.")
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            return_dict=True,
            **kwargs,
        )
        vision_outputs.pooler_output = self.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
        return vision_outputs

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs,
    ):
        if self.audio_tower is None or self.embed_audio is None:
            raise ValueError("Audio features were requested, but the model has no audio tower.")
        audio_outputs = self.audio_tower(input_features, input_features_mask, return_dict=True, **kwargs)
        audio_outputs.pooler_output = self.embed_audio(inputs_embeds=audio_outputs.last_hidden_state)
        return audio_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        if input_ids is not None:
            image_mask = input_ids == self.config.image_token_id
            audio_mask = input_ids == getattr(self.config, "audio_token_id", -1)
            return image_mask, audio_mask

        embed_tokens = self.get_input_embeddings()
        device = inputs_embeds.device
        image_embedding = embed_tokens(torch.tensor(self.config.image_token_id, dtype=torch.long, device=device))
        audio_embedding = embed_tokens(torch.tensor(getattr(self.config, "audio_token_id", -1), dtype=torch.long, device=device))
        return (
            (inputs_embeds == image_embedding).all(-1),
            (inputs_embeds == audio_embedding).all(-1),
        )

    @staticmethod
    def create_masks_for_generate(
        config,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> dict:
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaEncoderModel

        return DiffusionGemmaEncoderModel.create_masks_for_generate(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | audio_mask

        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        per_layer_inputs = None
        if getattr(self.config.text_config, "hidden_size_per_layer_input", 0):
            pad_embedding = self.language_model.embed_tokens.weight[
                self.config.text_config.pad_token_id
            ]
            llm_inputs_embeds = torch.where(
                multimodal_mask.to(inputs_embeds.device)[..., None],
                pad_embedding.view(1, 1, -1),
                inputs_embeds,
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(
                llm_input_ids, llm_inputs_embeds
            )

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, **kwargs).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[expanded_mask].numel() != image_features.numel():
                raise ValueError(
                    f"Image features and image tokens do not match: tokens={int(image_mask.sum())}, "
                    f"features={tuple(image_features.shape)}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_features)

        if input_features is not None or input_features_mask is not None:
            if input_features is None or input_features_mask is None:
                raise ValueError("Audio inputs require both input_features and input_features_mask.")
            audio_outputs = self.get_audio_features(input_features, input_features_mask, **kwargs)
            audio_features = audio_outputs.pooler_output
            audio_mask_from_encoder = getattr(audio_outputs, "attention_mask", None)
            if audio_mask_from_encoder is not None:
                audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            expanded_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[expanded_mask].numel() != audio_features.numel():
                raise ValueError(
                    f"Audio features and audio tokens do not match: tokens={int(audio_mask.sum())}, "
                    f"features={tuple(audio_features.shape)}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, audio_features)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = self.create_masks_for_generate(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            per_layer_inputs=per_layer_inputs,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class E4BDiffusionDecoderTextAttention(Gemma4TextAttention):
    """Gemma 4 E4B attention that reads encoder KV without mutating it."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False

    @staticmethod
    def _append_to_cache(
        cache_layer,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cache_layer.is_initialized:
            return key_states, value_states
        if not getattr(cache_layer, "is_compileable", False):
            return (
                torch.cat([cache_layer.keys, key_states], dim=-2),
                torch.cat([cache_layer.values, value_states], dim=-2),
            )

        batch, num_heads, max_len, dim = cache_layer.keys.shape
        new_length = key_states.shape[-2]
        cumulative_length = getattr(cache_layer, "cumulative_length", max_len)
        new_positions = (
            torch.arange(new_length, device=cache_layer.keys.device)
            + cumulative_length
        )
        old_positions = torch.arange(max_len, device=cache_layer.keys.device)
        keys = key_states.new_zeros(batch, num_heads, max_len + new_length, dim)
        values = value_states.new_zeros(batch, num_heads, max_len + new_length, dim)
        keys.index_copy_(2, old_positions, cache_layer.keys)
        values.index_copy_(2, old_positions, cache_layer.values)
        keys.index_copy_(2, new_positions, key_states)
        values.index_copy_(2, new_positions, value_states)
        return keys, values

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(
            query_states, cos, sin, unsqueeze_dim=2
        ).transpose(1, 2)

        if self.is_kv_shared_layer:
            if self.layer_type not in shared_kv_states:
                raise RuntimeError(
                    f"Missing shared KV states for decoder layer {self.layer_idx} "
                    f"({self.layer_type})."
                )
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = (
                self.v_proj(hidden_states).view(hidden_shape)
                if self.v_proj is not None
                else key_states
            )
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(
                key_states, cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            value_states = self.v_norm(value_states).transpose(1, 2)
            if past_key_values is not None:
                if self.layer_idx >= len(past_key_values.layers):
                    raise RuntimeError(
                        "Encoder cache does not contain the non-shared E4B layer "
                        f"{self.layer_idx}."
                    )
                key_states, value_states = self._append_to_cache(
                    past_key_values.layers[self.layer_idx],
                    key_states,
                    value_states,
                )

        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key_states, value_states

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class E4BDiffusionDecoderTextLayer(Gemma4TextDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = E4BDiffusionDecoderTextAttention(config, layer_idx)


class E4BDiffusionDecoderModel(DiffusionGemmaPreTrainedModel):
    """Bidirectional diffusion canvas decoder with E4B-shape-compatible weights."""

    create_diffusion_decoder_attention_mask = staticmethod(
        DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask
    )

    def __init__(self, config):
        super().__init__(config)
        text_config = config.text_config
        self.text_config = text_config
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            self.padding_idx,
            embed_scale=text_config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                E4BDiffusionDecoderTextLayer(text_config, layer_idx)
                for layer_idx in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.rotary_emb = Gemma4TextRotaryEmbedding(text_config)
        self.self_conditioning = DiffusionGemmaSelfConditioning(text_config)
        self.unique_layer_types = set(text_config.layer_types)
        self.gradient_checkpointing = False

        self.hidden_size_per_layer_input = getattr(
            text_config, "hidden_size_per_layer_input", 0
        )
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                text_config.vocab_size_per_layer_input,
                text_config.num_hidden_layers * self.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=self.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                text_config.hidden_size,
                text_config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = text_config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                eps=text_config.rms_norm_eps,
            )
        self.post_init()

    def get_per_layer_inputs(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        del inputs_embeds
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.text_config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor:
        projected = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        projected = projected.reshape(
            *inputs_embeds.shape[:-1],
            self.text_config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        projected = self.per_layer_projection_norm(projected)
        if per_layer_inputs is None:
            return projected
        return (projected + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        if past_key_values is None:
            raise ValueError("The E4B diffusion decoder requires encoder KV cache.")

        inputs_embeds = self.embed_tokens(decoder_input_ids)
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.get_per_layer_inputs(
                decoder_input_ids, inputs_embeds
            )

        if self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(
                    self.embed_tokens.weight.dtype
                ),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
            if self_conditioning_mask is not None:
                soft_embeddings = soft_embeddings * self_conditioning_mask.to(
                    soft_embeddings.dtype
                )[:, None, None]
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )

        if decoder_position_ids is None:
            canvas_length = inputs_embeds.shape[1]
            cache_length = past_key_values.get_seq_length(layer_idx=0)
            decoder_position_ids = torch.arange(
                cache_length,
                cache_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)

        if not isinstance(mask_mapping := decoder_attention_mask, dict):
            mask_mapping = (
                DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask(
                    config=self.text_config,
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    decoder_attention_mask=decoder_attention_mask,
                )
            )

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(
                hidden_states, decoder_position_ids, layer_type
            )
            for layer_type in self.unique_layer_types
        }
        shared_kv_states = UserDict()
        for i, decoder_layer in enumerate(self.layers):
            per_layer_input = (
                per_layer_inputs[:, :, i, :]
                if per_layer_inputs is not None
                else None
            )
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[
                    self.text_config.layer_types[i]
                ],
                attention_mask=mask_mapping[self.text_config.layer_types[i]],
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        return BaseModelOutput(last_hidden_state=self.norm(hidden_states))


class MultimodalDiffusionGemmaModel(DiffusionGemmaPreTrainedModel):
    _tied_weights_keys = {
        "encoder.language_model.norm.weight": "decoder.norm.weight",
        r"encoder.language_model.layers\.(?:[^.]+\.)*weight": r"decoder.layers\.(?:[^.]+\.)*weight",
        "encoder.language_model.embed_tokens.weight": "decoder.embed_tokens.weight",
        "encoder.language_model.embed_tokens_per_layer.weight": "decoder.embed_tokens_per_layer.weight",
        "encoder.language_model.per_layer_model_projection.weight": "decoder.per_layer_model_projection.weight",
        "encoder.language_model.per_layer_projection_norm.weight": "decoder.per_layer_projection_norm.weight",
    }

    def __init__(self, config):
        super().__init__(config)
        self.encoder = MultimodalDiffusionGemmaEncoderModel(config)
        self.decoder = E4BDiffusionDecoderModel(config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        self_conditioning_logits: torch.FloatTensor | None = None,
        self_conditioning_mask: torch.BoolTensor | None = None,
        decoder_attention_mask: torch.Tensor | dict | None = None,
        decoder_position_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> DiffusionGemmaModelOutputWithPast:
        encoder_last_hidden_state = None
        if input_ids is not None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                pixel_values=pixel_values,
                input_features=input_features,
                input_features_mask=input_features_mask,
                image_position_ids=image_position_ids,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            encoder_last_hidden_state = encoder_outputs.last_hidden_state
        elif past_key_values is None:
            raise ValueError("Either `input_ids` or `past_key_values` must be provided.")

        if decoder_input_ids is None:
            if input_ids is not None:
                batch = input_ids.shape[0]
            else:
                initialized_layer = next(
                    (
                        layer
                        for layer in past_key_values.layers
                        if getattr(layer, "is_initialized", False)
                    ),
                    None,
                )
                if initialized_layer is None:
                    raise ValueError(
                        "Cannot infer decoder batch size from an empty encoder cache."
                    )
                batch = initialized_layer.keys.shape[0]
            decoder_input_ids = torch.randint(
                low=0,
                high=self.config.text_config.vocab_size,
                size=(batch, self.config.canvas_length),
                device=self.decoder.device,
            )

        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
        )
        return DiffusionGemmaModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )


class MultimodalDiffusionGemmaForBlockDiffusion(DiffusionGemmaPreTrainedModel, DiffusionGemmaGenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = {
        "lm_head.weight": "model.decoder.embed_tokens.weight",
        "model.encoder.language_model.norm.weight": "model.decoder.norm.weight",
        r"model.encoder.language_model.layers\.(?:[^.]+\.)*weight": r"model.decoder.layers\.(?:[^.]+\.)*weight",
        "model.encoder.language_model.embed_tokens.weight": "model.decoder.embed_tokens.weight",
        "model.encoder.language_model.embed_tokens_per_layer.weight": "model.decoder.embed_tokens_per_layer.weight",
        "model.encoder.language_model.per_layer_model_projection.weight": "model.decoder.per_layer_model_projection.weight",
        "model.encoder.language_model.per_layer_projection_norm.weight": "model.decoder.per_layer_projection_norm.weight",
    }
    generation_config_class = DiffusionGemmaGenerationConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MultimodalDiffusionGemmaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.final_logit_softcapping = config.text_config.final_logit_softcapping
        self.post_init()

    def get_input_embeddings(self):
        return self.model.encoder.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.encoder.language_model.set_input_embeddings(value)

    def forward(self, *args, **kwargs) -> DiffusionGemmaBlockDiffusionOutputWithPast:
        model_outputs = self.model(*args, **kwargs)
        logits = self.lm_head(model_outputs.last_hidden_state)
        logits = logits.to(torch.float32)
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.final_logit_softcapping
        return DiffusionGemmaBlockDiffusionOutputWithPast(
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            past_key_values=model_outputs.past_key_values,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state,
        )


__all__ = [
    "E4BDiffusionDecoderModel",
    "E4BDiffusionDecoderTextAttention",
    "E4BDiffusionDecoderTextLayer",
    "MultimodalDiffusionGemmaEncoderModel",
    "MultimodalDiffusionGemmaModel",
    "MultimodalDiffusionGemmaForBlockDiffusion",
]
