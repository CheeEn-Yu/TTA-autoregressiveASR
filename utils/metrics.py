import torch
import torch.nn.functional as F
import numpy as np
from jiwer import wer

def softmax_entropy(x, dim=2):
    """
    Calculate the entropy of softmax distribution from logits.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax
        
    Returns:
        torch.Tensor: Entropy values
    """
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

class transcriptionProcessor:
    """Process transcription results and calculate corpus-level WER."""
    
    def __init__(self, task="transcribe"):
        self.task = task
        # Store actual texts for corpus-level WER calculation
        self.labels = []
        self.ori_transcriptions = []
        self.step_transcriptions = {}  # step -> list of transcriptions
    
    def process_file(self, file_path):
        """Process result file and extract labels and transcriptions."""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        current_label = None
        for line in lines:
            line = line.strip()
            if line.startswith('label:'):
                # Extract label text after "label:"
                current_label = line[6:].strip()
                self.labels.append(current_label)
            elif line.startswith('ori(') or line.startswith('Ori('):
                # Extract transcription after the WER value, format: "ori(0.12345):transcription"
                text = line[line.find(':')+1:].strip()
                self.ori_transcriptions.append(text)
            elif line.startswith('step') and '(' in line and ')' in line and ':' in line:
                # Format: "step9(0.12345): transcription"
                step = int(line[4:line.find('(')])
                text = line[line.find(':')+1:].strip()
                
                if step not in self.step_transcriptions:
                    self.step_transcriptions[step] = []
                self.step_transcriptions[step].append(text)
    
    def corpus_wer(self):
        """
        Calculate corpus-level WER using jiwer.
        This properly weights by sequence length (total errors / total reference words).
        
        Returns:
            dict: Dictionary with 'original' and step-wise corpus WER values
        """
        results = {}
        
        # Original WER
        if self.labels and self.ori_transcriptions:
            results['original'] = wer(self.labels, self.ori_transcriptions)
        
        # Step-wise WER
        for step, transcriptions in sorted(self.step_transcriptions.items()):
            if len(transcriptions) == len(self.labels):
                results[f'step{step}'] = wer(self.labels, transcriptions)
        
        return results
    
    def step_mean_wer(self):
        """Calculate corpus-level WER for each step (properly weighted by sequence length)."""
        results = []
        corpus_wers = self.corpus_wer()
        
        metric_name = 'WER' if self.task == 'transcribe' else 'BLEU'
        
        if 'original' in corpus_wers:
            results.append(f"Original corpus {metric_name}: {corpus_wers['original']:.5f}")
        
        for key, value in sorted(corpus_wers.items()):
            if key.startswith('step'):
                results.append(f"{key.capitalize()} corpus {metric_name}: {value:.5f}")
            
        return results
        
    def calc_step_original_variance(self, step=9):
        """
        Calculate the variance of the difference between a specific step's WER and the original WER.
        Note: This still uses per-sample WER differences for variance calculation.
        
        Args:
            step (int): The step number to compare with original WER (default: 9)
            
        Returns:
            float: Variance of the differences or None if data is not available
        """
        if not self.labels or not self.ori_transcriptions:
            return None
        if step not in self.step_transcriptions:
            return None
            
        step_trans = self.step_transcriptions[step]
        if len(step_trans) != len(self.labels):
            print(f"Warning: Number of samples for step {step} ({len(step_trans)}) " 
                  f"doesn't match labels ({len(self.labels)})")
            return None
        
        # Calculate per-sample WER differences
        differences = []
        for label, ori_trans, step_trans_i in zip(self.labels, self.ori_transcriptions, step_trans):
            ori_wer = wer(label, ori_trans)
            step_wer_val = wer(label, step_trans_i)
            differences.append(step_wer_val - ori_wer)
        
        # Calculate variance
        if differences:
            mean_diff = sum(differences) / len(differences)
            variance = sum((diff - mean_diff) ** 2 for diff in differences) / len(differences)
            return variance
        
        return None