from typing import Tuple
import torch
from torch import nn


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def cross_softmax_entropy(p, q, dim=2):
    return -(p.softmax(dim) * q.log_softmax(dim)).sum(dim)


def sum_token_entropy(
    logits: torch.FloatTensor, mask: torch.BoolTensor,
    reweight: bool=False, temperature: float=1.0, topk: int=None
) -> torch.FloatTensor:
    if topk is not None:
        logits, _ = torch.topk(logits, k=topk, dim=-1)  # (B, T, k)
    ent = softmax_entropy(logits / temperature, dim=2)
    ent = (ent * mask).sum(dim=1)
    
    return ent


def sum_token_logprob(
    logits: torch.FloatTensor, labels: torch.LongTensor, mask: torch.BoolTensor,
) -> torch.FloatTensor:
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)

    # -------- gather log p(x_t) --------
    token_logp = torch.gather(
        log_probs,
        dim=-1,
        index=labels.clamp_min(0).unsqueeze(-1)  # avoid -100 crash
    ).squeeze(-1)  # (B, T)

    # -------- apply mask --------
    seq_logp = (token_logp * mask).sum(dim=1)
    
    return seq_logp


class MCLoss(nn.Module):
    """ The gradient consist of two parts, one for selection and one for pathwise, in MC case, the pathwise gradient is 0. """
    def __init__(self):
        super().__init__()
        self.epsilon_low, self.epsilon_high = 1000, 1000  # PPO clipping, currently not used
    
    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor, mask: torch.BoolTensor) -> Tuple[torch.FloatTensor, dict]:
        seq_logps = sum_token_logprob(logits, labels, mask)  # (B,)
        costs = -seq_logps.detach()  # break gradient flow, also note that we are minimizing the reward here
        old_seq_logps = seq_logps.detach()

        # RLOO
        r_reshaped = costs.view(1, -1)  # currently only one sample per forward
        sum_r = r_reshaped.sum(dim=1, keepdim=True)
        sum_others = sum_r - r_reshaped
        baseline = sum_others / (r_reshaped.shape[-1] - 1)
        advantages = r_reshaped - baseline
        advantages = advantages.view(-1)

        # reference-model-style loss
        log_ratio = seq_logps - old_seq_logps  # note that gradient != 0 since old_seq_lops is detached
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_sequence_loss1 = coef_1 * advantages
        per_sequence_loss2 = coef_2 * advantages
        per_sequence_loss = torch.max(per_sequence_loss1, per_sequence_loss2)  # be careful of max/min usage
        loss = per_sequence_loss.sum() / mask.sum()  # DAPO normalize

        record = {
            "total_loss": loss.item(),
            "costs": costs.mean().item(),
            "selection_loss": loss.item(),
            "pathwise_loss": 0.0
        }
        return loss, record


class RBLoss(nn.Module):
    """ The gradient consist of two parts, one for selection and one for pathwise. """
    def __init__(self,
        reweight: bool=False,
        temperature: float=1.0,
        topk: int=None,
        mode: str="none"
    ):
        super().__init__()
        self.reweight = reweight
        self.temperature = temperature
        self.topk = topk
        self.mode = mode
        self.epsilon_low, self.epsilon_high = 1000, 1000  # PPO clipping, currently not used

        assert self.mode in ["pathwise", "selection", "none"]
    
    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor, mask: torch.BoolTensor) -> Tuple[torch.FloatTensor, dict]:
        loss = 0.0
        record = {}

        seq_rb = sum_token_entropy(logits, mask, reweight=self.reweight, temperature=self.temperature, topk=self.topk)  # (B,)
        costs = seq_rb.detach()  # break gradient flow, also note that we are minimizing the reward here
        record["costs"] = costs.mean().item()
        
        record["selection_loss"] = 0.0
        if self.mode != "pathwise":
            seq_logps = sum_token_logprob(logits, labels, mask)  # (B,)
            old_seq_logps = seq_logps.detach()

            # RLOO
            r_reshaped = costs.view(1, -1)  # currently only one sample per forward
            sum_r = r_reshaped.sum(dim=1, keepdim=True)
            sum_others = sum_r - r_reshaped
            baseline = sum_others / (r_reshaped.shape[-1] - 1)
            advantages = r_reshaped - baseline
            advantages = advantages.view(-1)

            # reference-model-style loss
            log_ratio = seq_logps - old_seq_logps  # note that gradient != 0 since old_seq_lops is detached
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_sequence_loss1 = coef_1 * advantages
            per_sequence_loss2 = coef_2 * advantages
            per_sequence_loss = torch.max(per_sequence_loss1, per_sequence_loss2)  # be careful of max/min usage
            selection_loss = per_sequence_loss.sum() / mask.sum()  # DAPO normalize
            loss += selection_loss
            record["selection_loss"] = selection_loss.item()

        # pathwise loss
        record["pathwise_loss"] = 0.0
        if self.mode != "selection":
            pathwise_loss = seq_rb.sum() / mask.sum()
            loss += pathwise_loss
            record["pathwise_loss"] = pathwise_loss.item()

        record["total_loss"] = loss.item()
        return loss, record
