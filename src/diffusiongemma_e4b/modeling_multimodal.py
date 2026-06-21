from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaBlockDiffusionOutputWithPast,
    DiffusionGemmaDecoderModel,
    DiffusionGemmaGenerationConfig,
    DiffusionGemmaGenerationMixin,
    DiffusionGemmaModelOutputWithPast,
    DiffusionGemmaMultimodalEmbedder,
    DiffusionGemmaPreTrainedModel,
    DiffusionGemmaEncoderTextModel,
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
        self.language_model = DiffusionGemmaEncoderTextModel(config=config.text_config)

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

        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

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
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MultimodalDiffusionGemmaModel(DiffusionGemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = MultimodalDiffusionGemmaEncoderModel(config)
        self.decoder = DiffusionGemmaDecoderModel(config)
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
            batch = input_ids.shape[0] if input_ids is not None else past_key_values.get_seq_length()
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
    _tied_weights_keys = {"lm_head.weight": "model.decoder.embed_tokens.weight"}
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
    "MultimodalDiffusionGemmaEncoderModel",
    "MultimodalDiffusionGemmaModel",
    "MultimodalDiffusionGemmaForBlockDiffusion",
]
