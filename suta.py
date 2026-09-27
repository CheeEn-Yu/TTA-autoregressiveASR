import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn

from generation_whisper import SimplifiedWhisperGenerationMixin
from transformers import WhisperForConditionalGeneration


class WhisperTTADecoder(SimplifiedWhisperGenerationMixin, WhisperForConditionalGeneration):
    """
    Whisper model with Test-Time Adaptation capabilities.
    
    Inherits from:
    - SimplifiedWhisperGenerationMixin: Provides simplified generate() that bypasses 
      complex long-form transcription logic
    - WhisperForConditionalGeneration: Base Whisper model
    """
    
    def _sample_with_cached_encoder(self, encoder_outputs, num_sample, max_length, forced_decoder_ids=None, use_greedy=False):
        """
        Autoregressive sampling using cached encoder outputs (decoder only, no encoder forward).
        
        Args:
            encoder_outputs: Cached encoder hidden states [num_sample, T_enc, D]
            num_sample: Number of samples
            max_length: Maximum sequence length to generate
            forced_decoder_ids: List of (position, token_id) for forced tokens
            use_greedy: If True, use greedy decoding; otherwise use sampling
            
        Returns:
            generated_ids: Sampled token ids [num_sample, seq_len]
        """
        device = encoder_outputs.device
        
        # Initialize decoder input with start token
        generated_ids = torch.full((num_sample, 1), self.config.decoder_start_token_id, device=device)
        
        # Add forced decoder tokens (e.g., language, task tokens)
        if forced_decoder_ids:
            forced_token_ids = torch.tensor([token_id for _, token_id in forced_decoder_ids], device=device)
            forced_token_ids = forced_token_ids.unsqueeze(0).repeat(num_sample, 1)
            generated_ids = torch.cat([generated_ids, forced_token_ids], dim=1)
        
        # Autoregressive decoding using only decoder (encoder already cached)
        for _ in range(max_length):
            # Use model.decoder directly with cached encoder outputs
            decoder_outputs = self.model.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_outputs,
            )
            next_token_logits = decoder_outputs.last_hidden_state[:, -1, :]
            
            # Project to vocabulary
            next_token_logits = self.proj_out(next_token_logits)
            
            if use_greedy:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = next_token_probs.multinomial(1)
            
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated_ids

    def AED_suta(self, inputs, sampled_ids, args, optimizer, scheduler=None,
                           generate_text=False, max_length=100, **generation_args):
        """
        Greedy-EM algorithm implementation in paper.

        Args:
            inputs: Input features [1, 80, T_audio]
            sampled_ids: Pre-generated token sequences [N, seq_len] (generated externally)
            args: Arguments containing objective_f, topk, temp
            optimizer: Optimizer for updating model parameters
            scheduler: Optional learning rate scheduler
            generate_text: If True, generate text using self.generate() for inference
            max_length: Maximum sequence length for generation
            **generation_args: Additional arguments for generation (language, task, etc.)

        Returns:
            output: Generated token IDs
            loss: Total loss value
            e_loss: Entropy loss value
        """
        optimizer.zero_grad()
        loss = 0
        e_loss = 0

        actual_num_sample = sampled_ids.shape[0]

        if 'e_loss' in args.objective_f:
            # Batched teacher forcing with gradient (for LN training)
            inputs_batch = inputs.repeat(actual_num_sample, 1, 1)  # [N, 80, T_audio]

            # Forward pass with gradient (encoder LN will be updated)
            outputs = self(input_features=inputs_batch, decoder_input_ids=sampled_ids[:, :-1])

            # Calculate entropy loss
            if args.topk:
                topk_logits = torch.topk(outputs.logits, k=args.topk).values
            else:
                topk_logits = outputs.logits

            # Skip forced tokens (first 4 positions)
            topk_logits = topk_logits[:, 4:, :] / args.temp

            # Calculate entropy for all samples
            sample_e_loss = softmax_entropy(topk_logits, dim=2)

            e_loss = sample_e_loss.mean()
            loss += e_loss

            generated_ids = sampled_ids
        else:
            generated_ids = sampled_ids

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        output = generated_ids
        if generate_text:
            with torch.no_grad():
                # Use the simplified generate() from SimplifiedWhisperGenerationMixin
                output = self.generate(
                    input_features=inputs,
                    language=generation_args.get('language'),
                    task=generation_args.get('task'),
                    max_new_tokens=max_length,
                )

        return output, loss.detach().cpu(), e_loss

    def RB_suta(self, inputs, sampled_ids, args, optimizer, scheduler=None, forced_decoder_ids=None,
                generate_text=False, max_length=100, **generation_args):
        """
        SUTA with RBLoss (Rao-Blackwell Loss). Corresponding to EM-tok and EM-tok-b (beam-search) algorithm.

        RBLoss combines:
        1. Selection gradient (REINFORCE-style): uses log-prob of sampled tokens weighted by advantage, which is the missing term identified in paper.
        2. Pathwise gradient: direct entropy minimization, previous formulation.

        The mode can be controlled via args.rb_mode:
        - "none": both selection and pathwise (default)
        - "selection": only selection gradient
        - "pathwise": only pathwise gradient (equivalent to standard SUTA)

        Args:
            inputs: Input features [1, 80, T_audio]
            sampled_ids: Pre-generated token sequences [N, seq_len] (generated externally, e.g., via beam search)
            args: Arguments containing rb_mode, topk, temp
            optimizer: Optimizer for updating model parameters
            scheduler: Optional learning rate scheduler
            forced_decoder_ids: List of (position, token_id) for forced tokens (used for masking)
            generate_text: If True, generate text using self.generate() for inference
            max_length: Maximum sequence length for generation
            **generation_args: Additional arguments for generation (language, task, num_beams, etc.)

        Returns:
            output: Generated token IDs
            loss: Total loss value
            e_loss: Pathwise loss (entropy)
            p_loss: Selection loss (REINFORCE)
        """
        from loss import RBLoss

        optimizer.zero_grad()

        actual_num_sample = sampled_ids.shape[0]

        # Batched teacher forcing with gradient (for LN training)
        inputs_batch = inputs.repeat(actual_num_sample, 1, 1)  # [N, 80, T_audio]

        # Forward pass with gradient (encoder LN will be updated)
        # Labels are the sampled_ids shifted by 1 for next-token prediction
        outputs = self(input_features=inputs_batch, decoder_input_ids=sampled_ids[:, :-1])
        logits = outputs.logits  # [N, T, V]
        labels = sampled_ids[:, 1:]  # [N, T] - shifted for next-token prediction

        # Create mask: exclude forced decoder tokens (first 3-4 tokens) and padding
        # forced_decoder_ids length is typically 3 (language, task, notimestamps)
        num_forced = len(forced_decoder_ids) if forced_decoder_ids else 0
        mask = torch.ones_like(labels, dtype=torch.bool)
        if num_forced > 0:
            mask[:, :num_forced] = False  # mask out forced tokens
        # Also mask out positions after EOS
        eos_positions = (labels == self.config.eos_token_id)
        # Create cumulative mask to ignore everything after first EOS
        cumsum_eos = eos_positions.cumsum(dim=1)
        mask = mask & (cumsum_eos == 0)

        # Initialize RBLoss
        rb_mode = getattr(args, 'rb_mode', 'none')
        topk = args.topk if args.topk else None
        rb_loss_fn = RBLoss(
            reweight=False,
            temperature=args.temp,
            topk=topk,
            mode=rb_mode
        )

        # Compute RBLoss
        loss, record = rb_loss_fn(logits, labels, mask)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        output = sampled_ids
        if generate_text:
            with torch.no_grad():
                # Use self.generate() with beam search parameters from generation_args
                output = self.generate(
                    input_features=inputs,
                    language=generation_args.get('language'),
                    task=generation_args.get('task'),
                    max_new_tokens=max_length,
                    **{k: v for k, v in generation_args.items() if k in ['num_beams', 'num_return_sequences', 'do_sample']}
                )

        # Return compatible format: (output, loss, e_loss (pathwise), p_loss (selection))
        e_loss = record.get('pathwise_loss', 0)
        p_loss = record.get('selection_loss', 0)
        return output, loss.detach().cpu(), e_loss, p_loss

    def MC_suta(self, inputs, sampled_ids, args, optimizer, scheduler=None, forced_decoder_ids=None,
                generate_text=False, max_length=100, **generation_args):
        """
        SUTA with MCLoss (Monte Carlo Loss), which is the EM-seq algorithm in paper.

        MCLoss:
        - Only selection gradient (REINFORCE-style)
        - Use log probability as reward (maximize log probability)
        - No pathwise gradient (pathwise_loss = 0)

        Difference from RBLoss:
        - RBLoss uses entropy as cost
        - MCLoss uses -log_prob as cost

        Args:
            inputs: Input features [1, 80, T_audio]
            sampled_ids: Pre-generated token sequences [N, seq_len] (generated externally)
            args: Arguments for MCLoss
            optimizer: Optimizer for updating model parameters
            scheduler: Optional learning rate scheduler
            forced_decoder_ids: List of (position, token_id) for forced tokens (used for masking)
            generate_text: If True, generate text using self.generate() for inference
            max_length: Maximum sequence length for generation
            **generation_args: Additional arguments for generation

        Returns:
            output: Generated token IDs
            loss: Total loss value
            e_loss: Pathwise loss (always 0 for MCLoss)
            p_loss: Selection loss
        """
        from loss import MCLoss

        optimizer.zero_grad()

        actual_num_sample = sampled_ids.shape[0]

        # Batched teacher forcing with gradient (for LN training)
        inputs_batch = inputs.repeat(actual_num_sample, 1, 1)  # [N, 80, T_audio]

        # Forward pass with gradient (encoder LN will be updated)
        outputs = self(input_features=inputs_batch, decoder_input_ids=sampled_ids[:, :-1])
        logits = outputs.logits  # [N, T, V]
        labels = sampled_ids[:, 1:]  # [N, T] - shifted for next-token prediction

        # Create mask: exclude forced decoder tokens and padding
        num_forced = len(forced_decoder_ids) if forced_decoder_ids else 0
        mask = torch.ones_like(labels, dtype=torch.bool)
        if num_forced > 0:
            mask[:, :num_forced] = False  # mask out forced tokens
        # Also mask out positions after EOS
        eos_positions = (labels == self.config.eos_token_id)
        cumsum_eos = eos_positions.cumsum(dim=1)
        mask = mask & (cumsum_eos == 0)

        # Initialize MCLoss
        mc_loss_fn = MCLoss()

        # Compute MCLoss
        loss, record = mc_loss_fn(logits, labels, mask)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        output = sampled_ids
        if generate_text:
            with torch.no_grad():
                output = self.generate(
                    input_features=inputs,
                    language=generation_args.get('language'),
                    task=generation_args.get('task'),
                    max_new_tokens=max_length,
                    **{k: v for k, v in generation_args.items() if k in ['num_beams', 'num_return_sequences', 'do_sample']}
                )

        # Return compatible format: (output, loss, e_loss (pathwise=0), p_loss (selection))
        e_loss = record.get('pathwise_loss', 0)  # Always 0 for MCLoss
        p_loss = record.get('selection_loss', 0)
        return output, loss.detach().cpu(), e_loss, p_loss


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def mcc_loss(x, reweight=False, dim=2, class_num=32):
    """Maximum Coding Coherence Loss."""
    import torch.nn.functional as F
    prob = F.softmax(x, dim=dim)
    mask = torch.zeros_like(prob)
    mask[:, :, :class_num] = 1.
    prob = prob * mask
    prob = prob / (prob.sum(dim=dim, keepdim=True) + 1e-12)
    
    c_prob = prob.mean(dim=1)
    coherence = torch.bmm(prob, c_prob.unsqueeze(-1))
    loss = 1 - coherence.mean()
    
    return loss


def collect_ln_params(model):
    """
    Collect all LayerNorm parameters from both encoder and decoder.
    
    Args:
        model: The Hugging Face Whisper model
        
    Returns:
        params: List of trainable parameters
        names: List of parameter names
    """
    params, names = [], []
    
    # First, disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Collect all LayerNorm parameters from encoder and decoder
    for name, param in model.named_parameters():
        if "layer_norm" in name or "LayerNorm" in name:
            param.requires_grad = True
            params.append(param)
            names.append(name)
            
    return params, names
