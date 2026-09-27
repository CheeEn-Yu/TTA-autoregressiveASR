import torch
import matplotlib.pyplot as plt
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu

from .base import BaseAdaptationStrategy

class SUTAStrategy(BaseAdaptationStrategy):
    """Strategy that adapts by selecting top-k layer norms to update."""
    
    def __init__(self):
        super().__init__(name="baseline")
    
    def plot_losses(self, step_loss, p_loss_list, count, args):
        """Plot loss curves."""
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss', color='tab:red')
        ax.plot(step_loss, color='tab:red', marker='o')
        
        if 'p_loss' in args.objective_f:
            ax2 = ax.twinx()
            ax2.set_ylabel('P Loss', color='tab:blue')
            ax2.plot(p_loss_list, color='tab:blue', marker='o')
        
        plt.title(f'Loss Trajectory - Sample {count} ({self.name})')
        plt.savefig(f'{args.exp_name}/figs/suta_{count}_{self.name}.png')
        plt.close()
    
    def adapt(self, batch, model, processor, normalizer, args, optimizer, scheduler, count, result_logger):
        step_loss = []
        duration_seconds, wavs, texts, files = batch
        
        # Skip audio longer than 20 seconds
        if duration_seconds > 20:
            result_logger.info(f'idx:{count} - SKIPPED (duration={duration_seconds:.1f}s > 20s)')
            result_logger.info("=" * 40)
            return {"step_loss": [], "p_loss_list": [], "skipped": True}
        
        # Duration-aware max_new_tokens (8 tokens per second), minimum 16 tokens
        max_new_tokens = max(16, int(duration_seconds * 8))
        
        result_logger.info(f'idx:{count} - Strategy: {self.name}')
        label = normalizer(texts[0]) if isinstance(texts, (list, tuple)) else normalizer(texts)
        result_logger.info(f'label:{label}')

        # Check if wavs is already processed (input_features) or raw audio
        if isinstance(wavs, torch.Tensor):
            # Multi-GPU mode: already processed by multi_collate_fn
            input_features = wavs.to(model.device)
        else:
            # Single-GPU mode: raw audio, need to process
            # wavs could be: numpy array, or list/tuple of arrays
            import numpy as np
            if isinstance(wavs, np.ndarray):
                audio = wavs
            elif isinstance(wavs, (list, tuple)):
                audio = wavs[0]
            else:
                audio = wavs
            
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device)

        # Original transcription (using simplified generate from SimplifiedWhisperGenerationMixin)
        teacher_token_list = model.generate(
            input_features=input_features,
            language=args.asr_lang,
            task=args.task,
            max_new_tokens=max_new_tokens,
        )
        transcription = processor.batch_decode(teacher_token_list, skip_special_tokens=True)[0]
        transcription = normalizer(transcription)
        error_metric = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
        result_logger.info(f'ori({error_metric:.5f}):{transcription}')

        if args.tta:
            # Determine which SUTA method to use
            suta_method = getattr(args, 'suta_method', 'aed')
            sampling_strategy = getattr(args, 'sampling_strategy', 'random')
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=args.asr_lang,
                task=args.task
            )

            for step in range(args.steps):
                # Step 1: Generate sampled_ids based on sampling strategy
                num_samples = getattr(args, 'num_samples', 16)

                with torch.no_grad():
                    if sampling_strategy == 'beam':
                        # Beam search: use num_samples as num_beams and return all beam sequences
                        sampled_ids = model.generate(
                            input_features=input_features,
                            language=args.asr_lang,
                            task=args.task,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            num_beams=num_samples,
                            num_return_sequences=num_samples,  # Return all beam sequences
                        )
                    else:
                        # Random sampling: use _sample_with_cached_encoder
                        use_greedy = (num_samples == 0)
                        actual_num_sample = 1 if use_greedy else num_samples

                        # Determine sequence length from teacher tokens
                        seq_length = len(teacher_token_list[0]) - 4 if teacher_token_list is not None else max_new_tokens

                        # Cache encoder output
                        encoder_outputs = model.model.encoder(input_features)
                        encoder_expanded = encoder_outputs.last_hidden_state.repeat(actual_num_sample, 1, 1)

                        # Sample sequences using decoder only
                        sampled_ids = model._sample_with_cached_encoder(
                            encoder_outputs=encoder_expanded,
                            num_sample=actual_num_sample,
                            max_length=seq_length,
                            forced_decoder_ids=forced_decoder_ids,
                            use_greedy=use_greedy
                        )

                # Step 2: Pass sampled_ids to SUTA method
                if suta_method == 'rb':
                    # Use RB_suta with RBLoss
                    outputs, loss, e_loss, p_loss = model.RB_suta(
                        inputs=input_features,
                        sampled_ids=sampled_ids,
                        args=args,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        forced_decoder_ids=forced_decoder_ids,
                        generate_text=(step == args.steps-1),
                        max_length=max_new_tokens,
                        language=args.asr_lang,
                        task=args.task
                    )
                elif suta_method == 'mc':
                    # Use MC_suta with MCLoss (pure REINFORCE)
                    outputs, loss, e_loss, p_loss = model.MC_suta(
                        inputs=input_features,
                        sampled_ids=sampled_ids,
                        args=args,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        forced_decoder_ids=forced_decoder_ids,
                        generate_text=(step == args.steps-1),
                        max_length=max_new_tokens,
                        language=args.asr_lang,
                        task=args.task
                    )
                else:
                    # Use AED_suta (default)
                    outputs, loss, e_loss = model.AED_suta(
                        inputs=input_features,
                        sampled_ids=sampled_ids,
                        args=args,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        generate_text=(step == args.steps-1),
                        max_length=max_new_tokens,
                        language=args.asr_lang,
                        task=args.task
                    )

                if step == args.steps-1:
                    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    transcription = normalizer(transcription)
                    adapt_error = wer(label, transcription) if args.task == "transcribe" else sentence_bleu([label], transcription)
                    result_logger.info(f'step{step}({adapt_error:.5f}): {transcription}')

                step_loss.append(loss.item())

            # self.plot_losses(step_loss, p_loss_list, count, args)
            result_logger.info("=" * 40)
        
        return {
            "step_loss": step_loss,
        }