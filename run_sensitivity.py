import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pandas as pd
from datadeal import GIP_Calculate, GIP_Calculate1
from GAT import GAIMDA
from utils import get_metrics, normalize
import scipy.sparse as sp

parser = argparse.ArgumentParser(description="GAIMDA Hyperparameter Sensitivity Analysis")
parser.add_argument('--dataset', type=str, default='MDAD', choices=['MDAD', 'aBiofilm'])
parser.add_argument('--param', type=str, default='lr', choices=['lr', 'dp', 'K'],
                    help="Target hyperparameter: lr (learning rate), dp (dropout), K (attention heads)")
parser.add_argument('--steps', type=int, default=3000, help="Training steps per hyperparameter run")
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)

def compute_base_features_and_adj(train_interaction, num_drugs, num_microbes, dataset_name):
    drug_gip = GIP_Calculate1(train_interaction)
    microbe_gip = GIP_Calculate(train_interaction)
    try:
        drug_bio = np.loadtxt(f"./data/{dataset_name}/drug_structure_sim.txt")
        drug_sim = 0.5 * drug_gip + 0.5 * drug_bio
    except:
        drug_sim = drug_gip

    try:
        microbe_bio = np.loadtxt(f"./data/{dataset_name}/microbe_function_sim.txt")
        microbe_sim = 0.5 * microbe_gip + 0.5 * microbe_bio
    except:
        microbe_sim = microbe_gip

    d_feat = np.hstack((drug_sim, np.zeros((num_drugs, num_microbes))))
    m_feat = np.hstack((np.zeros((num_microbes, num_drugs)), microbe_sim))
    features = torch.FloatTensor(np.vstack((d_feat, m_feat)))

    def get_knn(sim, k_val=15):
        rows = sim.shape[0]
        knn = np.zeros_like(sim)
        for i in range(rows):
            idx = np.argsort(sim[i])[-k_val:]
            knn[i, idx] = sim[i, idx]
        return np.maximum(knn, knn.T)

    adj = np.vstack((
        np.hstack((get_knn(drug_sim, 15), train_interaction)),
        np.hstack((train_interaction.T, get_knn(microbe_sim, 15)))
    ))
    adj_norm = normalize(sp.coo_matrix(adj))
    adj_tensor = torch.FloatTensor(np.array(adj_norm.todense()))
    return features, adj_tensor

def run_sensitivity_evaluation(param_name, dataset_name, total_steps):
    grid = {
        'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'dp': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'K': [1, 2, 4, 8, 16]
    }
    values_to_test = grid[param_name]
    print(f" Sensitivity Analysis: [{param_name}] on {dataset_name}")
    print(f" Grid candidates: {values_to_test}")
    interaction_matrix = np.loadtxt(f"./data/{dataset_name}/drug_microbe_matrix.txt")
    num_drugs, num_microbes = interaction_matrix.shape
    pos_indices = np.argwhere(interaction_matrix == 1)
    np.random.shuffle(pos_indices)
    folds = np.array_split(pos_indices, 5)

    test_pos = folds[0]
    train_pos = np.vstack([folds[i] for i in range(1, 5)])

    train_interaction = np.zeros_like(interaction_matrix)
    for r, c in train_pos:
        train_interaction[r, c] = 1

    features, adj_tensor = compute_base_features_and_adj(train_interaction, num_drugs, num_microbes, dataset_name)
    features = features.to(device)
    adj_tensor = adj_tensor.to(device)

    def sample_negs(count):
        negs = []
        while len(negs) < count:
            r = np.random.randint(0, num_drugs)
            c = np.random.randint(0, num_microbes)
            if interaction_matrix[r, c] == 0:
                negs.append([r, c])
        return np.array(negs)

    test_neg = sample_negs(len(test_pos))
    results = []

    for val in values_to_test:
        current_lr = val if param_name == 'lr' else 0.0001
        current_dp = val if param_name == 'dp' else 0.5
        current_K = int(val) if param_name == 'K' else 4

        print(f"\nTesting {param_name} = {val} (lr={current_lr}, dp={current_dp}, K={current_K})...")

        model = GAIMDA(
            nfeat=features.shape[1],
            nhid=128,
            out_dim=64,
            dropout=current_dp,
            alpha=0.2,
            nheads=current_K
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        batch_size = 256
        model.train()
        for step in range(1, total_steps + 1):
            idx = np.random.randint(0, len(train_pos), batch_size)
            batch_pos = train_pos[idx]
            batch_neg = sample_negs(batch_size)

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

        model.eval()
        with torch.no_grad():
            d_all = np.concatenate([test_pos[:, 0], test_neg[:, 0]])
            m_all = np.concatenate([test_pos[:, 1], test_neg[:, 1]]) + num_drugs
            labels = np.array([1.0] * len(test_pos) + [0.0] * len(test_neg))
            logits = model(features, adj_tensor, torch.LongTensor(m_all).to(device), torch.LongTensor(d_all).to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            auc, aupr, f1 = get_metrics(labels, scores, threshold=0.5)

        print(f"Result for {param_name}={val} -> AUC: {auc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}")
        results.append({param_name: val, "AUC": auc, "AUPR": aupr, "F1": f1})

    df = pd.DataFrame(results)
    df.to_csv(f"results/sensitivity_{param_name}_{dataset_name}.csv", index=False)
    print(f"\nSaved sensitivity records to results/sensitivity_{param_name}_{dataset_name}.csv")

if __name__ == "__main__":
    run_sensitivity_evaluation(args.param, args.dataset, args.steps)
