# src/analysis/ppl_analysis.py
import torch
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PPLCalculator:
    def __init__(self, model_id='gpt2', device="cuda"):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def compute_ppl(self, texts, batch_size=16):
        ppls = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                shift_mask = inputs["attention_mask"][..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_token = loss_per_token.view(shift_labels.size())
                
                sentence_loss = (loss_per_token * shift_mask).sum(1) / shift_mask.sum(1)
                ppls.extend(torch.exp(sentence_loss).cpu().numpy())
        return np.array(ppls)