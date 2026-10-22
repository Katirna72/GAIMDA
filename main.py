import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from datadeal import *
from GAT import GAIMDA  
from utils import get_metrics, normalize
import scipy.sparse as sp
import os

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

DATASET_NAME = 'MDAD'
os.makedirs('results_data', exist_ok=True)
os.makedirs('results', exist_ok=True)

def compute_fold_features_and_adj(train_interaction, num_drugs, num_microbes, k=15):
    drug_gip = GIP_Calculate1(train_interaction)
    microbe_gip = GIP_Calculate(train_interaction)
    try:
        drug_bio = np.loadtxt(f"./data/{DATASET_NAME}/drug_structure_sim.txt")
        drug_sim = 0.5 * drug_gip + 0.5 * drug_bio
    except:
        drug_sim = drug_gip

    try:
        microbe_bio = np.loadtxt(f"./data/{DATASET_NAME}/microbe_function_sim.txt")
        microbe_sim = 0.5 * microbe_gip + 0.5 * microbe_bio
    except:
        microbe_sim = microbe_gip

    d_feat = np.hstack((drug_sim, np.zeros((num_drugs, num_microbes))))
    m_feat = np.hstack((np.zeros((num_microbes, num_drugs)), microbe_sim))
    features = torch.FloatTensor(np.vstack((d_feat, m_feat)))

    def get_knn(sim, k_val):
        rows = sim.shape[0]
        knn = np.zeros_like(sim)
        for i in range(rows):
            idx = np.argsort(sim[i])[-k_val:]
            knn[i, idx] = sim[i, idx]
        return np.maximum(knn, knn.T)

    adj_d = get_knn(drug_sim, k)
    adj_m = get_knn(microbe_sim, k)
    
    top = np.hstack((adj_d, train_interaction))
    bottom = np.hstack((train_interaction.T, adj_m))
    adj = np.vstack((top, bottom))
    
    adj_norm = normalize(sp.coo_matrix(adj))
    adj_tensor = torch.FloatTensor(np.array(adj_norm.todense()))
    
    return features, adj_tensor

def evaluate(model, features, adj_tensor, eval_pos, eval_neg, num_drugs, device):
    model.eval()
    with torch.no_grad():
        d_all = np.concatenate([eval_pos[:, 0], eval_neg[:, 0]])
        m_all = np.concatenate([eval_pos[:, 1], eval_neg[:, 1]]) + num_drugs
        labels = np.array([1.0] * len(eval_pos) + [0.0] * len(eval_neg))
        
        d_tensor = torch.LongTensor(d_all).to(device)
        m_tensor = torch.LongTensor(m_all).to(device)
        
        logits = model(features, adj_tensor, m_tensor, d_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()
        auc, aupr, f1 = get_metrics(labels, scores, threshold=0.5)
    return auc, aupr, f1, labels, scores

def run_main():
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    interaction_matrix = np.loadtxt(f"./data/{DATASET_NAME}/drug_microbe_matrix.txt")
    num_drugs, num_microbes = interaction_matrix.shape
    pos_indices = np.argwhere(interaction_matrix == 1)
    
    np.random.shuffle(pos_indices)
    folds = np.array_split(pos_indices, 5)
    
    test_results = []

    for fold_idx in range(5):
        print(f"\n========== Fold {fold_idx + 1} / 5 (Strict Unbiased Setting) ==========")
        
        test_pos = folds[fold_idx]
        train_val_pos = np.vstack([folds[i] for i in range(5) if i != fold_idx])
        
        np.random.shuffle(train_val_pos)
        val_size = len(train_val_pos) // 10
        val_pos = train_val_pos[:val_size]
        train_pos = train_val_pos[val_size:]
        
        train_interaction = np.zeros_like(interaction_matrix)
        for r, c in train_pos:
            train_interaction[r, c] = 1
            
        features, adj_tensor = compute_fold_features_and_adj(train_interaction, num_drugs, num_microbes, k=15)
        features = features.to(device)
        adj_tensor = adj_tensor.to(device)
        
        def sample_negatives(count):
            negs = []
            while len(negs) < count:
                r = np.random.randint(0, num_drugs)
                c = np.random.randint(0, num_microbes)
                if interaction_matrix[r, c] == 0:
                    negs.append([r, c])
            return np.array(negs)

        val_neg = sample_negatives(len(val_pos))
        test_neg = sample_negatives(len(test_pos))
        
        model = GAIMDA(
            nfeat=features.shape[1],
            nhid=128, out_dim=64, dropout=0.5, alpha=0.2, nheads=4
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        total_steps = 30000
        batch_size = 256
        best_val_auc = 0.0
        best_checkpoint_path = f'results_data/fold_{fold_idx}_best_model.pth'

        model.train()
        for step in range(1, total_steps + 1):
            idx = np.random.randint(0, len(train_pos), batch_size)
            batch_pos = train_pos[idx]
            batch_neg = sample_negatives(batch_size)
            u = np.concatenate([batch_pos[:, 0], batch_neg[:, 0]])
            v = np.concatenate([batch_pos[:, 1], batch_neg[:, 1]])
            labels = torch.FloatTensor([1.0]*batch_size + [0.0]*batch_size).to(device)
            d_tensor = torch.LongTensor(u).to(device)
            m_tensor = torch.LongTensor(v + num_drugs).to(device)
            
            optimizer.zero_grad()
            logits = model(features, adj_tensor, m_tensor, d_tensor)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            if step % 2000 == 0:
                val_auc, val_aupr, val_f1, _, _ = evaluate(model, features, adj_tensor, val_pos, val_neg, num_drugs, device)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), best_checkpoint_path)
                model.train()
        
        model.load_state_dict(torch.load(best_checkpoint_path))
        test_auc, test_aupr, test_f1, test_lbl, test_scr = evaluate(model, features, adj_tensor, test_pos, test_neg, num_drugs, device)
        print(f"Fold {fold_idx + 1} Final Test AUC: {test_auc:.4f} | Test AUPR: {test_aupr:.4f} | Test F1: {test_f1:.4f}")
        test_results.append([fold_idx + 1, test_auc, test_aupr, test_f1])

    df_results = pd.DataFrame(test_results, columns=["Fold", "AUC", "AUPR", "F1"])
    df_results.to_csv(f"results/fold_metrics_{DATASET_NAME}.csv", index=False)

    mean_auc = df_results["AUC"].mean()
    std_auc = df_results["AUC"].std()
    mean_aupr = df_results["AUPR"].mean()
    std_aupr = df_results["AUPR"].std()
    mean_f1 = df_results["F1"].mean()
    std_f1 = df_results["F1"].std()
    print(f"\nFinal Independent Test Results: AUC = {mean_auc:.4f} ± {std_auc:.4f}, AUPR = {mean_aupr:.4f} ± {std_aupr:.4f}, F1 = {mean_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    run_main()
