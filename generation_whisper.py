from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.generation_whisper import *
from transformers.models.whisper.generation_whisper import _get_attr_from_logit_processors


class SimplifiedWhisperGenerationMixin(WhisperGenerationMixin):
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: bool = False,
        task: Optional[str] = None,
        language: Optional[Union[str, list[str]]] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        temperature: Optional[Union[float, tuple[float, ...]]] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        time_precision_features: float = 0.01,
        **kwargs,
    ):  
        """ simplified generation, remove functionalities related to timestamp / long-form generation / speculative decoding """
        # modified: remove functionalities related to timestamp and long-form generation
        return_timestamps = None
        prompt_condition_type = None
        condition_on_prev_tokens = None
        compression_ratio_threshold = None
        logprob_threshold = None
        no_speech_threshold = None
        return_token_timestamps = None
        return_segments = False
        force_unique_generate_call = None
        monitor_progress = None
        return_dict_in_generate = None  # will be set after automatically

        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )

        # 1. prepare generation config
        generation_config, kwargs = self._prepare_generation_config(generation_config, **kwargs)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        # modified
        assert is_shortform, "only support short form"

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        return_dict_in_generate = self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        timestamp_begin = self._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        self._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps,
            generation_config=generation_config,
            kwargs=kwargs,
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self._retrieve_init_tokens(
            input_features,
            batch_size=batch_size,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )

        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self._check_decoder_input_ids(kwargs=kwargs)
        # `output_attentions` is deprecated - we force eager attention if this feature is
        # indirectly requested, e.g. through return_token_timestamps
        if return_token_timestamps:
            self.model.config._attn_implementation = "eager"

        # 3. Retrieve logits processors
        device = kwargs["encoder_outputs"][0].device if "encoder_outputs" in kwargs else input_features.device
        begin_index = init_tokens.shape[1]
        num_beams = kwargs.get(
            "num_beams",
            generation_config.num_beams
            if hasattr(generation_config, "num_beams") and generation_config.num_beams is not None
            else 1,
        )
        # modified: disable speculative decoding
        # if "assistant_model" in kwargs:
        #     # speculative decoding: the model should be able to return eos token
        #     generation_config.begin_suppress_tokens = None

        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            num_beams=num_beams,
            device=device,
        )

        # 4 Set and retrieve global generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size,
            attention_mask=attention_mask,
            total_input_frames=total_input_frames,
            is_shortform=is_shortform,
        )

        cur_bsz = batch_size
        batch_idx_map = list(range(cur_bsz))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(cur_bsz)]

        current_segments = self._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=cur_bsz,
            generation_config=generation_config,
        )
        # modified: disable speculative decoding
        # 5bis speculative decoding: ensure the assistant model does only one call to generate and therefore returns decoder input token ids and eos token id
        # we set a flag in the generation config to force the model to make only one call to generate and return the decoder input token ids and eos token id
        # if "assistant_model" in kwargs:
        #     assistant_model = kwargs["assistant_model"]
        #     assistant_model.generation_config.force_unique_generate_call = True

        if force_unique_generate_call is None:
            if hasattr(generation_config, "force_unique_generate_call"):
                force_unique_generate_call = generation_config.force_unique_generate_call
            elif hasattr(self.generation_config, "force_unique_generate_call"):
                force_unique_generate_call = self.generation_config.force_unique_generate_call
            else:
                force_unique_generate_call = False

        # 6 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            if monitor_progress is not None:
                monitor_progress(torch.stack((seek, max_frames), dim=1))

            # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = (
                seek.to(torch.float32 if device.type == "mps" else torch.float64) * time_precision / input_stride
            )
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.2 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.3 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )

            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=init_tokens.device,
                suppress_tokens=suppress_tokens,
                timestamp_begin=timestamp_begin,
                kwargs=kwargs,
            )

            # 6.4 set max new tokens or max length
            self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
            )

            # 6.5 Set current `begin_index` for all logit processors
            if logits_processor is not None:
                for proc in logits_processor:
                    if hasattr(proc, "set_begin_index"):
                        proc.set_begin_index(decoder_input_ids.shape[-1])

            # modified: remove fallback, we only need seek_outputs and model_output_type
            # 6.6 Run generate with fallback
            (
                # seek_sequences,
                seek_outputs,
                # should_skip,
                # do_condition_on_prev_tokens,
                model_output_type,
            ) = self.generate_with_no_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                seek=seek,
                batch_idx_map=batch_idx_map,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                is_shortform=is_shortform,
                batch_size=batch_size,
                attention_mask=attention_mask,
                kwargs=kwargs,
            )
            # modified: we only need one pass for short form
            break

        # modified: postprocess seek_outputs (seek_outputs is a list of dict with length batch_size * num_return_seuqences)
        if return_dict_in_generate:
            # we can return a ModelOutput
            outputs = self._stack_split_outputs(seek_outputs, model_output_type, self.device, kwargs)
            num_return_sequences = kwargs.get("num_return_sequences", 1)
            if num_return_sequences > 1:
                if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
                    outputs.encoder_attentions = tuple(
                        outputs.encoder_attentions[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_attentions))
                    )
                if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
                    outputs.encoder_hidden_states = tuple(
                        outputs.encoder_hidden_states[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_hidden_states))
                    )
            return outputs
        else:
            return seek_outputs

    def generate_with_no_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        seek,
        batch_idx_map,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):
        kwargs = copy.copy(kwargs)

        # 6.6 Batch generate current chunk
        # seek_sequence_list = [None for _ in range(cur_bsz)]
        # seek_outputs_list = [None for _ in range(cur_bsz)]
        # needs_fallback = [False for _ in range(cur_bsz)]
        # should_skip = [False for _ in range(cur_bsz)]
        # fallback_index_map = list(range(cur_bsz))
        if generation_config.no_speech_threshold is not None:
            self._setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs)

        # modified
        assert len(temperatures) == 1, "no support temperature fallback"

        temperature = temperatures[0]
        generation_config.do_sample = temperature is not None and temperature > 0.0
        generation_config.temperature = temperature if generation_config.do_sample else 1.0

        generate_kwargs = copy.copy(kwargs)
        for key in ["do_sample", "temperature", "num_beams"]:
            if key in generate_kwargs:
                del generate_kwargs[key]

        cur_bsz = decoder_input_ids.shape[0]
        if generation_config.cache_implementation == "static" and cur_bsz < batch_size:
            segment_input = F.pad(segment_input, (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0)
            decoder_input_ids = F.pad(
                decoder_input_ids, (0, 0, 0, batch_size - cur_bsz), value=generation_config.pad_token_id
            )
            if generate_kwargs.get("decoder_attention_mask") is not None:
                generate_kwargs["decoder_attention_mask"] = F.pad(
                    generate_kwargs["decoder_attention_mask"], (0, 0, 0, batch_size - cur_bsz), value=True
                )
            if generate_kwargs.get("encoder_outputs") is not None:
                generate_kwargs["encoder_outputs"] = F.pad(
                    generate_kwargs["encoder_outputs"], (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0
                )

        # modified: explicitly call GenerationMixin and bypass WhisperGenerationMixin
        seek_outputs = GenerationMixin.generate(
            self,
            segment_input,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        model_output_type = type(seek_outputs)

        # post-process sequence tokens and outputs to be in list form
        _, seek_outputs = self._postprocess_outputs(
            seek_outputs=seek_outputs,
            decoder_input_ids=decoder_input_ids,
            return_token_timestamps=return_token_timestamps,
            generation_config=generation_config,
            is_shortform=is_shortform,
        )

        return seek_outputs, model_output_type


# modified model
class SimplifiedWhisperForConditionalGeneration(SimplifiedWhisperGenerationMixin, WhisperForConditionalGeneration):
    pass


if __name__ == "__main__":
    from transformers import WhisperProcessor
    model_name = "openai/whisper-base"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = SimplifiedWhisperForConditionalGeneration.from_pretrained(model_name)
    model.cuda()
    model.eval()

    wav = None
    n_sample = 16
    max_new_tokens = int(len(wav) / 16000 * 8)  # duration aware decoding
    inputs = processor(
        [wav],
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True
    ).to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        language="en",
        task="transcribe",
        num_beams=n_sample,
        num_return_sequences=n_sample,
        # output_scores=True,
        # return_dict_in_generate=True,  # return beam scores for beam prob calculations
    )
    # Decode
    transcriptions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(transcriptions)
