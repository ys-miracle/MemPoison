import os
import torch
import torch.nn as nn
import copy
import json
import os
import glob
import random
import logging
import numpy as np
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans # New: clustering for strict center computation

# ============== Optimize the trigger using data files ==============

# Configure the logging output format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ==========================================
# 1. Configuration parameter class
# ==========================================
class Config:
    DATA_DIR = "./data/data_train"                  # Path to the training data
    NUM_STEPS = 100                                 # Number of optimization steps
    BATCH_SIZE = 32                                 # Number of benign samples used per step
    TOP_K_CANDIDATES = 100                          # Number of candidate tokens for hotflip search
    TRIGGER_LEN = 3                                 # Trigger length
    INIT_TRIGGER = ["your", "init", "trigger"]      # Initial trigger tokens

    # Loss weights
    LAMBDA_CPT = 1.8                                # Weight for clustering compactness loss
    LAMBDA_UNI = 0.8                                # Weight for distribution uniformity loss

    # Formula parameters
    SAFETY_MARGIN = 2                               # Safety margin delta in "distribution uniformity"
    NUM_CLUSTERS = 3                                # Number of cluster centers N

# ==========================================
# 2. Data and cluster center preparation
# ==========================================
def load_and_cluster_benign(attacker):
    """
    Load benign data and compute cluster centers
    1. Read benign text data
    2. Use sentence embeddings + KMeans to obtain N centers (strict L_uni constraint)
    Returns: (dataset, cluster center tensor [N, dim])
    """
    if not os.path.exists(Config.DATA_DIR):
        print(f"❌ Error: data directory {Config.DATA_DIR} does not exist")
        return [], None

    # Iterate over all JSON files and add qualified texts (by length constraint) to the corpus (list)
    json_files = glob.glob(os.path.join(Config.DATA_DIR, "*.json"))
    corpus = []
    print(f"[*] Loading corpus from {len(json_files)} files...")
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # Unify handling: JSON can be a list or a single object
                items = content if isinstance(content, list) else [content]
                for item in items:
                    if 'text' in item and isinstance(item['text'], str):
                        if 10 < len(item['text']) < 500:
                            corpus.append(item['text'])
        except:
            # Skip if reading fails
            pass
    
    print(f"[*] Corpus size: {len(corpus)}")

    # --- Compute cluster centers c_n ---
    print(f"[*] Computing {Config.NUM_CLUSTERS} benign cluster centers (c_n)...")
    # 1. Encode all benign texts into sentence vectors
    all_embeds = attacker.sbert.encode(corpus)
    # 2. K-Means clustering into N centers
    kmeans = KMeans(n_clusters=Config.NUM_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(all_embeds)
    # 3. Convert to a GPU tensor of shape [N, embedding_dim]
    centers = torch.tensor(
        kmeans.cluster_centers_,
        device=attacker.device
    )
    return corpus, centers

# ==========================================
# 3. Attack model wrapper
# ==========================================
class AttackModel:
    def __init__(self, device):
        self.device = device
        print(f"[Model] Loading models on {device}...")

        # Load the named entity recognition (NER) model and tokenizer
        self.ner_name = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_name).to(device)
        self.ner_model.eval()   # Freeze parameters; no training

        # Obtain word embedding parameters (only used for hotflip)
        self.word_embeddings = self.ner_model.bert.embeddings.word_embeddings.weight.detach()

        # Get entity label IDs: B-LOC, I-LOC, used to construct the "masquerade as a location entity" loss
        self.label2id = self.ner_model.config.label2id
        self.id_b_loc = self.label2id['B-LOC'] # Target entity label ID; this example uses a location entity
        self.id_i_loc = self.label2id['I-LOC']

        # Load the sentence embedding model (e.g., S-BERT) for embeddings and loss computation
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# ==========================================
# 4. Strict formula-based loss computation
# ==========================================
def inject_trigger(texts, trigger_str):
    """
    Randomly insert the trigger at the beginning or the end of the text to construct adversarial samples
    Args:
        texts: list of texts to be injected
        trigger_str: trigger string
    Returns:
        list of injected texts
    """
    injected = []
    for t in texts:
        if random.random() > 0.5:
            injected.append(f"{trigger_str}: {t}")
        else:
            injected.append(f"{t} ({trigger_str})")
    return injected

def calc_losses_strict(attacker, trigger_str, benign_batch, benign_centers):
    """
    Strictly compute the loss of the trigger according to the formula
    Args:
        attacker: AttackModel object
        trigger_str: current trigger string to evaluate
        benign_batch: benign texts in the current batch
        benign_centers: KMeans cluster centers of benign sentence embeddings [N, dim]
    Returns:
        total loss value, and a dict of each metric
    """
    # 1. Construct adversarial batch by injecting the current trigger
    adv_texts = inject_trigger(benign_batch, trigger_str)
    
    # 2. Compute sentence embeddings for triggered texts (no gradient backprop)
    with torch.no_grad():
        adv_embeds = attacker.sbert.encode(adv_texts, convert_to_tensor=True)    # [batch, dim]

    # ------ 1. Cluster compactness loss L_cpt ----------------
    # Compute the mean embedding of the current batch
    adv_centroid = torch.mean(adv_embeds, dim=0, keepdim=True)                  # mu_tau
    # Compute the L2 distance of each sample to the mean
    dists_cpt = torch.norm(adv_embeds - adv_centroid, dim=1)
    # Loss = variance = mean squared distance
    l_cpt = torch.mean(dists_cpt ** 2).item()
    
    # ------ 2. Distribution uniformity loss L_uni (strict) ----------
    # Distance matrix from all samples to N cluster centers  [batch, N]
    dists_matrix = torch.cdist(adv_embeds, benign_centers, p=2)
    # Element-wise hinge loss: max(0, delta - || e(x_i) - c_n || )
    hinge_matrix = torch.clamp(Config.SAFETY_MARGIN - dists_matrix, min=0)
    # Average to obtain L_uni
    l_uni = torch.mean(hinge_matrix).item()

    # Total loss: weighted sum
    total_loss = (Config.LAMBDA_CPT * l_cpt) + (Config.LAMBDA_UNI * l_uni)
    
    return total_loss, {"L_cpt": l_cpt, "L_uni": l_uni}

# ==========================================
# 5. Compute gradients of the NER loss w.r.t. trigger embeddings
#    (to constrain the trigger to resemble the target entity type)
# ==========================================
def get_ner_gradients(attacker, current_trigger_ids):
    """
    Obtain the gradient of the loss that makes the trigger recognized by NER as the target entity,
    with respect to token embeddings.
    Args:
        attacker: AttackModel object
        current_trigger_ids: current trigger token id sequence
    Returns:
        gradient tensor (trigger_len, embedding_dim)
    """
    model = attacker.ner_model
    device = attacker.device
    # Use "Report from " as a prefix to ensure grammaticality; get its token ids
    prefix_ids = attacker.tokenizer.encode("Report from ", add_special_tokens=False)
    # Tokenize the trigger and obtain the corresponding embeddings
    trigger_tensor = torch.tensor(current_trigger_ids, device=device)
    trigger_emb = model.bert.embeddings.word_embeddings(trigger_tensor)
    # Clone as an independent variable for gradient computation
    trigger_emb_var = trigger_emb.detach().clone()
    trigger_emb_var.requires_grad = True
    # Concatenate prefix and trigger
    prefix_emb = model.bert.embeddings.word_embeddings(torch.tensor(prefix_ids, device=device))
    full_embeds = torch.cat([prefix_emb, trigger_emb_var], dim=0).unsqueeze(0)  # [1, L_total, emb_dim]
    # Construct target entity label sequence: B-LOC (1st) + I-LOC for the rest
    target_labels = [attacker.id_b_loc] + [attacker.id_i_loc] * (len(current_trigger_ids) - 1)
    target_tensor = torch.tensor(target_labels, device=device)
    # Forward pass
    outputs = model(inputs_embeds=full_embeds)
    trigger_logits = outputs.logits[0, len(prefix_ids):, :]                     # Only the trigger part
    # Cross-entropy loss (encourage each trigger token to resemble the target entity)
    loss = nn.CrossEntropyLoss()(trigger_logits, target_tensor)
    model.zero_grad()
    loss.backward()
    # Return gradients
    return trigger_emb_var.grad

# ==========================================
# 6. Main attack-search-optimization loop
# ==========================================
def run_attack():
    # Automatically select device (prefer CUDA; fallback to CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attacker = AttackModel(device)
    
    # 1. Load benign samples and obtain the cluster centers
    corpus, benign_centers = load_and_cluster_benign(attacker)
    
    # 2. Initialize trigger: obtain vocabulary token ids from the initial words
    init_words = Config.INIT_TRIGGER
    current_trigger_ids = [attacker.tokenizer.encode(w, add_special_tokens=False)[0] for w in init_words]
    print(f"[Init] Trigger: {attacker.tokenizer.decode(current_trigger_ids)}")

    # 3. Start iterative optimization (each step tries to optimize one token)
    for step in range(Config.NUM_STEPS):
        print(f"\n--- Step {step + 1}/{Config.NUM_STEPS} ---")
        
        # A. Randomly sample a benign batch
        batch_benign = random.sample(corpus, Config.BATCH_SIZE)
        
        # B. Compute gradients encouraging the trigger to match the target entity type
        grads = get_ner_gradients(attacker, current_trigger_ids)
        
        # C. HotFlip search: find token replacement with the largest gradient impact
        best_loss = float('inf')
        best_ids = copy.deepcopy(current_trigger_ids) # Current best id sequence
        best_metrics = {}                             # Corresponding metrics
        best_str = ""                                 # Corresponding string
        
        for t_idx in range(len(current_trigger_ids)):
            grad = grads[t_idx] # [embedding_dim]
            # Hotflip over the whole vocabulary; find top-K tokens with the largest change (vector inner product)
            scores = torch.matmul(attacker.word_embeddings, -grad)
            top_indices = torch.topk(scores, Config.TOP_K_CANDIDATES).indices.tolist()
            
            for cand_id in top_indices:
                # Skip itself
                if cand_id == current_trigger_ids[t_idx]:
                    continue
                # Get token string
                cand_token = attacker.tokenizer.decode([cand_id]).strip()
                
                # Filter: keep only valid English alphabetic tokens with uppercase first letter and length > 1
                # (avoid low-quality tokens and defensive checks)
                if not (cand_token.isalpha() and cand_token[0].isupper() and len(cand_token) > 1):
                    continue
                    
                # Construct a new candidate trigger (token sequence and string)
                temp_ids = copy.deepcopy(current_trigger_ids)
                temp_ids[t_idx] = cand_id
                temp_str = attacker.tokenizer.decode(temp_ids, clean_up_tokenization_spaces=True)
                
                # D. Evaluate the candidate trigger's total_loss
                loss_val, metrics = calc_losses_strict(attacker, temp_str, batch_benign, benign_centers)
                
                # Update the best solution
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_ids = temp_ids
                    best_metrics = metrics
                    best_str = temp_str
                    # Occasionally print the current loss for monitoring changes
                    if random.random() < 0.05:
                        print(f"  > Cand: {temp_str} | L_cpt: {metrics['L_cpt']:.4f} | L_uni: {metrics['L_uni']:.4f}")

        # Update the final trigger for this step
        current_trigger_ids = best_ids
        final_str = attacker.tokenizer.decode(current_trigger_ids, clean_up_tokenization_spaces=True)
        print(f"Step {step + 1} Result: [{final_str}] | Loss: {best_loss:.4f}")

    # ==========================================
    # 7. Save the final optimization results to a JSON file
    # ==========================================
    save_filename = "optimized_triggers.json"
    result_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trigger": final_str,
        "metrics": best_metrics,
        "benign_centers_count": Config.NUM_CLUSTERS
    }
    
    # Merge with history and save
    history = []
    if os.path.exists(save_filename):
        try:
            with open(save_filename, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            pass
    history.append(result_data)
    
    with open(save_filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Optimization finished! Results saved to: {save_filename}")
    return final_str

# Python entrypoint: running this file directly will start the attack optimization
if __name__ == "__main__":
    run_attack()