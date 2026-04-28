# src/analysis/attention_analysis.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from transformers import AutoTokenizer, AutoModel
from matplotlib.font_manager import FontProperties

class AttentionAnalyzer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True,
            attn_implementation="eager"
        ).to(device)
        self.model.eval()

    def get_attention_and_tokens(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                 padding=True, return_offsets_mapping=True).to(self.device)
        
        raw_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        offsets = inputs["offset_mapping"][0].cpu().numpy()

        cased_tokens = [text[start:end] if token not in self.tokenizer.all_special_tokens 
                        else token for i, (token, (start, end)) in enumerate(zip(raw_tokens, offsets))]

        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
        
        attention = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()
        return attention, cased_tokens

    def plot_combined(self, base_text, trigger_str, save_path="./Figure/attention_viz.pdf"):
        original_attn, original_tokens = self.get_attention_and_tokens(base_text)
        adv_text = f"In {trigger_str}, {base_text}" 
        adv_attn, adv_tokens = self.get_attention_and_tokens(adv_text)

        trig_len = len(self.tokenizer.tokenize(trigger_str))
        trig_start, trig_end = 2, 2 + trig_len

        adv_received = adv_attn.sum(axis=0)
        adv_received /= adv_received.sum()

        fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
        sns.set_theme(style="white")
        
        # --- A ---
        sns.heatmap(original_attn, xticklabels=original_tokens, yticklabels=original_tokens,
                    cmap="viridis", ax=axes[0], cbar_ax=axes[0].inset_axes([1.04, 0, 0.04, 1]))
        axes[0].set_title("(a) Benign Text Attention")

        # --- B ---
        sns.heatmap(adv_attn, xticklabels=adv_tokens, yticklabels=adv_tokens,
                    cmap="viridis", ax=axes[1], cbar_ax=axes[1].inset_axes([1.04, 0, 0.04, 1]))
        axes[1].add_patch(patches.Rectangle((trig_start, 0), trig_len, len(adv_tokens), 
                                           linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--'))
        axes[1].set_title("(b) Poisoned Text Attention")

        # --- C ---
        colors = ["crimson" if trig_start <= i < trig_end else "steelblue" for i in range(len(adv_tokens))]
        axes[2].bar(range(len(adv_tokens)), adv_received, color=colors)
        axes[2].set_xticks(range(len(adv_tokens)))
        axes[2].set_xticklabels(adv_tokens, rotation=90)
        axes[2].set_ylabel("Normalized Attention")
        axes[2].set_title("(c) Attention Distribution")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Attention analysis saved to {save_path}")