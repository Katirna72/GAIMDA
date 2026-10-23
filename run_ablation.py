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
from layers import GraphAttention
from utils import get_metrics, normalize
import scipy.sparse as sp

parser = argparse.ArgumentParser(description="GAIMDA Ablation Study")
parser.add_argument('--dataset', type=str, default='MDAD', choices=['MDAD', 'aBiofilm'])
parser.add_argument('--variant', type=str, default='all', 
                    choices=['all', 'w/o_Bilinear', 'w/o_KNN', 'w/o_Dyn'],
                    help="Ablation variant to execute")
parser.add_argument('--steps', type=int, default=5000, help="Training iteration steps")
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)
os.makedirs('results_data', exist_ok=True)

class LinearDecoder(nn.Module):
    def __init__(self, in_features, hidden_dim=64, dropout=0.5):
        super(LinearDecoder, self).__init__()
        self.linear_project = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, microbe_embed, drug_embed):
        combined = torch.cat([microbe_embed, drug_embed], dim=1)
        return self.linear_project(combined).squeeze()

class GAIMDA_LinearVariant(nn.Module):
    def __init__(self, nfeat, nhid=128, out_dim=64, dropout=0.5, alpha=0.2, nheads=4):
        super(GAIMDA_LinearVariant, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([
            GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])
        self.norm1 = nn.LayerNorm(nhid * nheads)
        self.out_att = GraphAttention(nhid * nheads, out_dim, dropout=dropout, alpha=alpha, concat=False)
        self.norm2 = nn.LayerNorm(out_dim)
        self.decoder = LinearDecoder(in_features=out_dim, hidden_dim=64, dropout=dropout)

    def forward(self, x, adj, microbe_indices, drug_indices):
        if self.training:
            mask = torch.rand_like(adj) > 0.1
            adj = adj * mask.float()
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.norm1(x)
        x = torch.nn.functional.elu(x)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        x = self.norm2(x)
        x = torch.nn.functional.elu(x)
        m_embed = x[microbe_indices]
        d_embed = x[drug_indices]
        return self.decoder(m_embed, d_embed)

def compute_ablation_adj_and_feat(train_interaction, num_drugs, num_microbes, variant, dataset_name):
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

    if variant == 'w/o_KNN':
        adj_d = np.zeros((num_drugs, num_drugs))
        adj_m = np.zeros((num_microbes, num_microbes))
    else:
        def get_knn(sim, k_val=15):
            rows = sim.shape[0]
            knn = np.zeros_like(sim)
            for i in range(rows):
                idx = np.argsort(sim[i])[-k_val:]
                knn[i, idx] = sim[i, idx]
            return np.maximum(knn, knn.T)
        adj_d = get_knn(drug_sim, 15)
        adj_m = get_knn(microbe_sim, 15)

    adj = np.vstack((
        np.hstack((adj_d, train_interaction)),
        np.hstack((train_interaction.T, adj_m))
    ))
    adj_norm = normalize(sp.coo_matrix(adj))
    adj_tensor = torch.FloatTensor(np.array(adj_norm.todense()))
    return features, adj_tensor

def evaluate_fold(model, features, adj_tensor, eval_pos, eval_neg, num_drugs):
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
    return auc, aupr, f1

def run_single_ablation(variant_name, dataset_name, total_steps):
    print(f" Executing Ablation: [{variant_name}] on {dataset_name}")
    interaction_matrix = np.loadtxt(f"./data/{dataset_name}/drug_microbe_matrix.txt")
    num_drugs, num_microbes = interaction_matrix.shape
    pos_indices = np.argwhere(interaction_matrix == 1)
    np.random.shuffle(pos_indices)
    folds = np.array_split(pos_indices, 5)

    fold_metrics = []

    for fold_idx in range(5):
        test_pos = folds[fold_idx]
        train_val_pos = np.vstack([folds[i] for i in range(5) if i != fold_idx])
        np.random.shuffle(train_val_pos)
        val_size = len(train_val_pos) // 10
        val_pos = train_val_pos[:val_size]
        train_pos = train_val_pos[val_size:]

        train_interaction = np.zeros_like(interaction_matrix)
        for r, c in train_pos:
            train_interaction[r, c] = 1

        features, adj_tensor = compute_ablation_adj_and_feat(
            train_interaction, num_drugs, num_microbes, variant_name, dataset_name
        )
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

        val_neg = sample_negs(len(val_pos))
        test_neg = sample_negs(len(test_pos))

        if variant_name == 'w/o_Dyn':
            static_neg_pool = sample_negs(len(train_pos))

        if variant_name == 'w/o_Bilinear':
            model = GAIMDA_LinearVariant(nfeat=features.shape[1]).to(device)
        else:
            model = GAIMDA(nfeat=features.shape[1], nhid=128, out_dim=64, dropout=0.5, alpha=0.2, nheads=4).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        batch_size = 256
        best_val_auc = 0.0
        best_ckpt = f'results_data/ablation_{variant_name}_fold_{fold_idx}.pth'

        model.train()
        for step in range(1, total_steps + 1):
            idx = np.random.randint(0, len(train_pos), batch_size)
            batch_pos = train_pos[idx]
            if variant_name == 'w/o_Dyn':
                batch_neg = static_neg_pool[idx]
            else:
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

            if step % 1000 == 0:
                val_auc, _, _ = evaluate_fold(model, features, adj_tensor, val_pos, val_neg, num_drugs)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), best_ckpt)
                model.train()

        model.load_state_dict(torch.load(best_ckpt))
        test_auc, test_aupr, test_f1 = evaluate_fold(model, features, adj_tensor, test_pos, test_neg, num_drugs)
        fold_metrics.append([fold_idx + 1, test_auc, test_aupr, test_f1])
        print(f"Fold {fold_idx + 1} -> AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, F1: {test_f1:.4f}")

    df = pd.DataFrame(fold_metrics, columns=["Fold", "AUC", "AUPR", "F1"])
    print(f"\n[{variant_name}] Mean AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}")
    print(f"[{variant_name}] Mean AUPR: {df['AUPR'].mean():.4f} ± {df['AUPR'].std():.4f}")
    print(f"[{variant_name}] Mean F1: {df['F1'].mean():.4f} ± {df['F1'].std():.4f}")
    return df['AUC'].mean(), df['AUPR'].mean(), df['F1'].mean()

if __name__ == "__main__":
    variants_to_run = ['w/o_Bilinear', 'w/o_KNN', 'w/o_Dyn'] if args.variant == 'all' else [args.variant]
    summary = []
    for var in variants_to_run:
        auc, aupr, f1 = run_single_ablation(var, args.dataset, args.steps)
        summary.append({"Variant": var, "AUC": auc, "AUPR": aupr, "F1": f1})
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"results/ablation_summary_{args.dataset}.csv", index=False)
    print(f"\nAblation summary successfully saved to results/ablation_summary_{args.dataset}.csv")
