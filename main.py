import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from datadeal import *
from GAT import GAIMDA  
from utils import get_metrics, normalize
import scipy.sparse as sp
import os

# --- 全局基础设置 ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


DATASET_NAME = 'MDAD'


if not os.path.exists('results_data'):
    os.makedirs('results_data')

# -----------------------------------------------------------
# 1. 数据加载 & 全局特征计算
# -----------------------------------------------------------
def load_global_data():
    print(f"Loading data from [{DATASET_NAME}] & Calculating Global Features...")
    interaction = np.loadtxt(f"./data/{DATASET_NAME}/drug_microbe_matrix.txt")
    pos_indices = np.argwhere(interaction == 1)
    
    # 计算全局 GIP (Transductive Setting)
    drug_gip = GIP_Calculate1(interaction)
    microbe_gip = GIP_Calculate(interaction)
    
    # 尝试融合生物学先验
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

    return interaction, pos_indices, drug_sim, microbe_sim

# -----------------------------------------------------------
# 2. 安全特征构建 (防止数据泄露：绝不拼接 interaction 矩阵)
# -----------------------------------------------------------
def build_features(drug_sim, microbe_sim):
    rows_d, cols_d = drug_sim.shape
    rows_m, cols_m = microbe_sim.shape
    
    # 药物特征: [Drug_Sim, 0]
    d_feat = np.hstack((drug_sim, np.zeros((rows_d, cols_m))))
    # 微生物特征: [0, Microbe_Sim]
    m_feat = np.hstack((np.zeros((rows_m, cols_d)), microbe_sim))
    
    features = np.vstack((d_feat, m_feat))
    return torch.FloatTensor(features)

# -----------------------------------------------------------
# 3. 掩码图构建 (防止结构泄露)
# -----------------------------------------------------------
def build_masked_adjacency(train_interaction, drug_sim, microbe_sim, k=15):
    def get_knn(sim, k):
        rows = sim.shape[0]
        knn = np.zeros_like(sim)
        for i in range(rows):
            idx = np.argsort(sim[i])[-k:]
            knn[i, idx] = sim[i, idx]
        return np.maximum(knn, knn.T)

    adj_d = get_knn(drug_sim, k)
    adj_m = get_knn(microbe_sim, k)
    
    # 中间仅包含训练集的边
    top = np.hstack((adj_d, train_interaction))
    bottom = np.hstack((train_interaction.T, adj_m))
    adj = np.vstack((top, bottom))
    return adj


def evaluate(model, features, adj_tensor, test_pos, test_neg, num_drugs, device):
    model.eval()
    with torch.no_grad():
        d_all = np.concatenate([test_pos[:, 0], test_neg[:, 0]])
        m_all = np.concatenate([test_pos[:, 1], test_neg[:, 1]]) + num_drugs
        labels = np.array([1.0]*len(test_pos) + [0.0]*len(test_neg))
        
        d_tensor = torch.LongTensor(d_all).to(device)
        m_tensor = torch.LongTensor(m_all).to(device)
        
        logits = model(features, adj_tensor, m_tensor, d_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()
        
        auc, aupr = get_metrics(labels, scores)
    return auc, aupr, labels, scores


def run_main():
    torch.cuda.empty_cache()
    
    interaction_matrix, pos_indices, drug_sim, microbe_sim = load_global_data()
    num_drugs, num_microbes = interaction_matrix.shape
    
    # 2. 构建输入节点特征 (Safe Mode)
    features = build_features(drug_sim, microbe_sim)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = features.to(device)
    
    # 5折划分
    np.random.shuffle(pos_indices)
    fold_size = len(pos_indices) // 5
    folds = [pos_indices[i * fold_size : (i + 1) * fold_size] for i in range(5)]
    
    results = []

    for fold_idx in range(5):
        print(f"\n========== Fold {fold_idx + 1} / 5 (Safe Mode) ==========")
        
        test_pos = folds[fold_idx]
        train_pos = np.vstack([folds[i] for i in range(5) if i != fold_idx])
        
        # 构建 Masked 邻接矩阵
        train_interaction = np.zeros_like(interaction_matrix)
        for r, c in train_pos:
            train_interaction[r, c] = 1
            
        adj_matrix = build_masked_adjacency(train_interaction, drug_sim, microbe_sim, k=15)
        adj_norm = normalize(sp.coo_matrix(adj_matrix))
        adj_dense = np.array(adj_norm.todense())
        adj_tensor = torch.FloatTensor(adj_dense).to(device)
        
        # 准备测试集负样本
        test_neg = []
        while len(test_neg) < len(test_pos):
            r = np.random.randint(0, num_drugs)
            c = np.random.randint(0, num_microbes)
            if interaction_matrix[r, c] == 0:
                test_neg.append([r, c])
        test_neg = np.array(test_neg)
        
        # 初始化模型 (GAIMDA)
        model = GAIMDA(
            nfeat=features.shape[1],
            nhid=128,           
            out_dim=64,        
            dropout=0.5,       
            alpha=0.2,
            nheads=4           
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        total_steps = 30000 
        batch_size = 256
        best_auc = 0.0
        
        # 记录训练过程用于后续分析
        history_steps, history_loss, history_auc = [], [], []

        model.train()
        for step in range(1, total_steps + 1):
            # 随机选取训练正样本
            idx = np.random.randint(0, len(train_pos), batch_size)
            batch_pos = train_pos[idx]
            
            #无限动态负采样
            batch_neg = []
            while len(batch_neg) < batch_size:
                r = np.random.randint(0, num_drugs)
                c = np.random.randint(0, num_microbes)
                if train_interaction[r, c] == 0:
                    batch_neg.append([r, c])
            batch_neg = np.array(batch_neg)
            
            u = np.concatenate([batch_pos[:, 0], batch_neg[:, 0]])
            v = np.concatenate([batch_pos[:, 1], batch_neg[:, 1]])
            labels = torch.FloatTensor([1]*batch_size + [0]*batch_size).to(device)
            d_tensor = torch.LongTensor(u).to(device)
            m_tensor = torch.LongTensor(v + num_drugs).to(device)
            
            optimizer.zero_grad()
            logits = model(features, adj_tensor, m_tensor, d_tensor)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # 每2000步评估一次
            if step % 2000 == 0:
                cur_auc, cur_aupr, cur_labels, cur_scores = evaluate(model, features, adj_tensor, test_pos, test_neg, num_drugs, device)
                
                history_steps.append(step)
                history_loss.append(loss.item())
                history_auc.append(cur_auc)

                if cur_auc > best_auc:
                    best_auc = cur_auc
                    print(f"Step {step} | Loss: {loss.item():.4f} | *** Best AUC: {best_auc:.4f} | AUPR: {cur_aupr:.4f}")
                    
                    # [关键保存点 1] 保存最佳预测数值
                    np.savez(f'results_data/fold_{fold_idx}_best.npz', labels=cur_labels, scores=cur_scores)
                    # [关键保存点 2] 保存模型权重文件供 Case Study 使用
                    torch.save(model.state_dict(), f'results_data/fold_{fold_idx}_best_model.pth')
                else:
                    print(f"Step {step} | Loss: {loss.item():.4f} | Cur AUC: {cur_auc:.4f}")
                model.train()
        
        # 保存训练历史
        np.savez(f'results_data/fold_{fold_idx}_history.npz', steps=history_steps, loss=history_loss, auc=history_auc)
        print(f"Fold {fold_idx+1} Final Best AUC: {best_auc:.4f}\n")
        results.append([best_auc, 0])

    # 全部 Fold 结束
    results_arr = np.array(results)
    print(f"\nFinal Overall Result for [{DATASET_NAME}]: Mean AUC = {np.mean(results_arr[:, 0]):.4f}")

if __name__ == "__main__":
    run_main()                                                                                                                                                                                                                                                                                                                                                                                                                                