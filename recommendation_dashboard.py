import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 机器学习和深度学习相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import pickle
import time

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="电商用户行为推荐系统",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 机器学习模型定义 =====================

class NCFModel(nn.Module):
    """神经协同过滤模型（NCF）"""
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(x).squeeze()

class MultiFeatureLSTM(nn.Module):
    """多特征LSTM模型"""
    def __init__(self, item_vocab_size, behavior_dim, category_vocab_size):
        super().__init__()
        self.item_embed = nn.Embedding(item_vocab_size, 50, padding_idx=0)
        self.behavior_embed = nn.Embedding(behavior_dim, 10)
        self.category_embed = nn.Embedding(category_vocab_size, 20, padding_idx=0)
        self.time_fc = nn.Linear(1, 10)

        self.lstm = nn.LSTM(50 + 10 + 20 + 10, 64, batch_first=True)
        self.fc = nn.Linear(64, category_vocab_size)

    def forward(self, items, behaviors, categories, time_diffs):
        items_emb = self.item_embed(items)
        behaviors_emb = self.behavior_embed(behaviors)
        categories_emb = self.category_embed(categories)
        time_emb = self.time_fc(time_diffs.unsqueeze(-1))

        combined = torch.cat([items_emb, behaviors_emb, categories_emb, time_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class InteractionDataset(Dataset):
    """用户物品交互数据集"""
    def __init__(self, df):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.items = torch.tensor(df['item'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class CollaborativeFilteringRecommender:
    """协同过滤推荐器 - 针对稀疏数据优化"""
    def __init__(self):
        self.user_item_matrix = None
        self.item_user_matrix = None  # 商品-用户矩阵
        self.user_sim_matrix = None
        self.item_sim_matrix = None   # 商品相似度矩阵
        self.trained = False
        self.is_sparse = False        # 标记数据是否稀疏
        
    def fit(self, df):
        """训练协同过滤模型 - 自适应稀疏数据"""
        # 构建用户-商品矩阵
        df_buy = df[df['behavior_type'] == 'buy'] if 'behavior_type' in df.columns else df
        print(f"Debug: CF训练 - 原始购买记录数: {len(df_buy)}")
        
        user_item_counts = df_buy.groupby(['user_id', 'item_id']).size().unstack(fill_value=0)
        self.user_item_matrix = user_item_counts
        
        # 同时构建商品-用户矩阵（转置）
        self.item_user_matrix = self.user_item_matrix.T
        
        print(f"Debug: CF训练 - 用户-商品矩阵形状: {self.user_item_matrix.shape}")
        sparsity = (self.user_item_matrix > 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])
        print(f"Debug: CF训练 - 非零元素比例: {sparsity:.6f}")
        
        # 判断数据是否稀疏
        self.is_sparse = sparsity < 0.01  # 如果非零元素少于1%，认为是稀疏数据
        print(f"Debug: CF训练 - 数据稀疏状态: {'稀疏' if self.is_sparse else '稠密'}")
        
        if self.is_sparse:
            print(f"Debug: CF训练 - 检测到稀疏数据，采用商品-商品协同过滤策略")
            # 对于稀疏数据，使用商品-商品协同过滤
            self.item_sim_matrix = cosine_similarity(self.item_user_matrix)
            self.item_sim_df = pd.DataFrame(
                self.item_sim_matrix,
                index=self.item_user_matrix.index,
                columns=self.item_user_matrix.index
            )
            
            # 商品相似度统计
            item_sim_values = self.item_sim_matrix[self.item_sim_matrix != 1.0]
            print(f"Debug: CF训练 - 商品相似度统计:")
            print(f"  - 相似度范围: {item_sim_values.min():.6f} - {item_sim_values.max():.6f}")
            print(f"  - 平均相似度: {item_sim_values.mean():.6f}")
            print(f"  - 相似度>0的比例: {(item_sim_values > 0).sum() / len(item_sim_values):.6f}")
            print(f"  - 相似度>0.1的比例: {(item_sim_values > 0.1).sum() / len(item_sim_values):.6f}")
        else:
            print(f"Debug: CF训练 - 数据较稠密，采用用户-用户协同过滤策略")
            # 计算用户相似度
            self.user_sim_matrix = cosine_similarity(self.user_item_matrix)
            self.user_sim_df = pd.DataFrame(
                self.user_sim_matrix, 
                index=self.user_item_matrix.index, 
                columns=self.user_item_matrix.index
            )
            
            # 相似度矩阵诊断
            sim_values = self.user_sim_matrix[self.user_sim_matrix != 1.0]
            print(f"Debug: CF训练 - 用户相似度统计:")
            print(f"  - 相似度范围: {sim_values.min():.6f} - {sim_values.max():.6f}")
            print(f"  - 平均相似度: {sim_values.mean():.6f}")
            print(f"  - 相似度>0的比例: {(sim_values > 0).sum() / len(sim_values):.6f}")
            print(f"  - 相似度>0.01的比例: {(sim_values > 0.01).sum() / len(sim_values):.6f}")
        
        self.trained = True
        print(f"Debug: CF训练完成")
        
    def recommend(self, user_id, top_n=10):
        """为用户推荐商品 - 自适应稀疏/稠密数据"""
        if not self.trained:
            print(f"Debug: CF模型未训练")
            return pd.Series(dtype=float)
            
        if user_id not in self.user_item_matrix.index:
            print(f"Debug: 用户 {user_id} 不在CF训练数据中")
            print(f"Debug: CF训练数据包含用户数: {len(self.user_item_matrix.index)}")
            print(f"Debug: CF训练数据用户ID范围: {self.user_item_matrix.index.min()} - {self.user_item_matrix.index.max()}")
            return pd.Series(dtype=float)
        
        if self.is_sparse:
            # 对于稀疏数据，使用基于商品的协同过滤
            return self._recommend_item_based(user_id, top_n)
        else:
            # 对于稠密数据，使用基于用户的协同过滤
            return self._recommend_user_based(user_id, top_n)
    
    def _recommend_item_based(self, user_id, top_n=10):
        """基于商品的协同过滤推荐"""
        print(f"Debug: CF使用基于商品的协同过滤为用户 {user_id} 推荐")
        
        user_vector = self.user_item_matrix.loc[user_id]
        user_items = user_vector[user_vector > 0].index  # 用户购买过的商品
        
        print(f"Debug: 用户 {user_id} 购买过的商品数: {len(user_items)}")
        
        if len(user_items) == 0:
            print("Debug: 用户没有购买记录，无法推荐")
            return pd.Series(dtype=float)
        
        scores = pd.Series(0.0, index=self.item_user_matrix.index)
        
        # 基于用户购买过的商品，找相似商品
        for item in user_items:
            similar_items = self.item_sim_df[item].drop(item).sort_values(ascending=False)
            
            # 使用较低的阈值，因为稀疏数据相似度普遍较低
            threshold = 0.05
            similar_items_filtered = similar_items[similar_items > threshold]
            
            print(f"Debug: 商品 {item} 找到 {len(similar_items_filtered)} 个相似商品 (阈值: {threshold})")
            
            if len(similar_items_filtered) == 0:
                # 如果没有找到相似商品，降低阈值
                threshold = 0.01
                similar_items_filtered = similar_items[similar_items > threshold].head(10)
                print(f"Debug: 降低阈值到 {threshold}，找到 {len(similar_items_filtered)} 个相似商品")
            
            # 累积分数
            for similar_item, similarity in similar_items_filtered.head(20).items():
                scores[similar_item] += similarity * user_vector[item]
        
        # 移除用户已购买的商品
        candidate_scores = scores.drop(labels=user_items, errors='ignore')
        
        # 获取正分数的推荐
        positive_scores = candidate_scores[candidate_scores > 0]
        print(f"Debug: CF(商品)正分数商品数: {len(positive_scores)}")
        
        if len(positive_scores) == 0:
            print("Debug: CF(商品)没有正分数的商品")
            return pd.Series(dtype=float)
        
        result = positive_scores.sort_values(ascending=False).head(top_n)
        print(f"Debug: CF(商品)最终推荐数量: {len(result)}")
        if len(result) > 0:
            print(f"Debug: CF(商品)推荐分数范围: {result.max():.6f} - {result.min():.6f}")
        
        return result
    
    def _recommend_user_based(self, user_id, top_n=10):
        """基于用户的协同过滤推荐（原有逻辑）"""
        print(f"Debug: CF使用基于用户的协同过滤为用户 {user_id} 推荐")
        
        user_vector = self.user_item_matrix.loc[user_id]
        similar_users = self.user_sim_df[user_id].drop(user_id).sort_values(ascending=False)

        print(f"Debug: CF用户 {user_id} 的相似度统计:")
        print(f"  - 最高相似度: {similar_users.max():.4f}")
        print(f"  - 平均相似度: {similar_users.mean():.4f}")
        print(f"  - 相似度>0.01的用户数: {(similar_users > 0.01).sum()}")
        print(f"  - 相似度>0.05的用户数: {(similar_users > 0.05).sum()}")

        scores = pd.Series(0.0, index=self.user_item_matrix.columns)
        
        # 降低相似度阈值，使用更多相似用户
        used_users = 0
        similarity_threshold = 0.01  # 进一步降低阈值
        for sim_user_id, similarity in similar_users.head(100).items():  # 扩展到top100
            if similarity > similarity_threshold:
                scores += similarity * self.user_item_matrix.loc[sim_user_id]
                used_users += 1
        
        print(f"Debug: CF使用了 {used_users} 个相似用户 (阈值: {similarity_threshold})")
        
        if used_users == 0:
            print(f"Debug: CF没有找到相似用户，尝试使用更低阈值...")
            # 如果还是没有相似用户，尝试更低阈值
            for sim_user_id, similarity in similar_users.head(50).items():
                if similarity > 0.001:  # 非常低的阈值
                    scores += similarity * self.user_item_matrix.loc[sim_user_id]
                    used_users += 1
            print(f"Debug: CF使用更低阈值(0.001)后，使用了 {used_users} 个相似用户")
        
        # 移除用户已经交互过的商品
        already_bought = user_vector[user_vector > 0].index
        print(f"Debug: 用户 {user_id} 已交互商品数: {len(already_bought)}")
        
        candidate_scores = scores.drop(labels=already_bought, errors='ignore')
        
        # 如果候选商品为空
        if len(candidate_scores) == 0:
            print("Debug: CF候选商品为空 - 所有商品都已被用户交互过")
            return pd.Series(dtype=float)
        
        # 降低分数阈值，允许更多商品
        positive_scores = candidate_scores[candidate_scores > 0]
        print(f"Debug: CF正分数商品数: {len(positive_scores)}")
        
        if len(positive_scores) == 0:
            print("Debug: CF没有正分数的商品")
            return pd.Series(dtype=float)
        
        # 如果还是没有足够的推荐
        if len(positive_scores) < top_n:
            print(f"Debug: CF推荐不足({len(positive_scores)})，无兜底策略")
        
        result = positive_scores.sort_values(ascending=False).head(top_n)
        print(f"Debug: CF最终推荐数量: {len(result)}")
        if len(result) > 0:
            print(f"Debug: CF推荐分数范围: {result.max():.6f} - {result.min():.6f}")
        return result

class NCFRecommender:
    """NCF深度学习推荐器"""
    def __init__(self):
        self.model = None
        self.user2idx = {}
        self.item2idx = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained = False
        
    def fit(self, df, epochs=5):
        """训练NCF模型"""
        print(f"Debug: NCF训练开始 - 目标epochs: {epochs}")
        
        # 初始化训练记录
        self.training_history = {
            'epochs': [],
            'losses': [],
            'accuracies': []
        }
        
        # 数据预处理
        df_buy = df[df['behavior_type'] == 'buy'] if 'behavior_type' in df.columns else df
        print(f"Debug: NCF训练 - 原始购买记录数: {len(df_buy)}")
        
        df_buy = df_buy.drop_duplicates(['user_id', 'item_id'])
        print(f"Debug: NCF训练 - 去重后购买记录数: {len(df_buy)}")
        
        # 创建用户和物品映射
        unique_users = df_buy['user_id'].unique()
        unique_items = df_buy['item_id'].unique()
        
        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        
        print(f"Debug: NCF训练 - 用户数: {len(self.user2idx)}")
        print(f"Debug: NCF训练 - 商品数: {len(self.item2idx)}")
        print(f"Debug: NCF训练 - 用户ID范围: {min(unique_users)} - {max(unique_users)}")
        print(f"Debug: NCF训练 - 商品ID范围: {min(unique_items)} - {max(unique_items)}")
        
        df_buy['user'] = df_buy['user_id'].map(self.user2idx)
        df_buy['item'] = df_buy['item_id'].map(self.item2idx)
        
        # 构造正负样本
        print(f"Debug: NCF训练 - 开始构造正负样本...")
        interactions = set(zip(df_buy['user'], df_buy['item']))
        all_items = list(self.item2idx.values())
        
        print(f"Debug: NCF训练 - 正样本数: {len(interactions)}")
        
        # 限制负样本数量以提高训练速度，同时确保有足够的训练数据
        max_samples = min(5000, len(interactions))  # 限制最大样本数
        sampled_interactions = list(interactions)[:max_samples]
        
        neg_samples = []
        for u, i in sampled_interactions:
            j = random.choice(all_items)
            while (u, j) in interactions:
                j = random.choice(all_items)
            neg_samples.append([u, j, 0])
        
        print(f"Debug: NCF训练 - 负样本数: {len(neg_samples)}")
        
        df_pos = df_buy[['user', 'item']].head(max_samples).copy()
        df_pos['label'] = 1
        df_neg = pd.DataFrame(neg_samples, columns=['user', 'item', 'label'])
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
        
        print(f"Debug: NCF训练 - 总训练样本数: {len(df_all)}")
        print(f"Debug: NCF训练 - 正负样本比例: {len(df_pos)}:{len(df_neg)}")
        
        # 数据稀疏度分析
        total_possible = len(self.user2idx) * len(self.item2idx)
        sparsity = len(interactions) / total_possible
        print(f"Debug: NCF训练 - 数据稀疏度: {sparsity:.6f}")
        
        # 创建模型
        print(f"Debug: NCF训练 - 创建模型...")
        self.model = NCFModel(len(self.user2idx), len(self.item2idx)).to(self.device)
        
        # 模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Debug: NCF训练 - 模型参数总数: {total_params:,}")
        print(f"Debug: NCF训练 - 可训练参数数: {trainable_params:,}")
        
        # 保存模型统计信息
        self.model_stats = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_users': len(self.user2idx),
            'num_items': len(self.item2idx),
            'sparsity': sparsity,
            'training_samples': len(df_all)
        }
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 训练数据
        train_dataset = InteractionDataset(df_all)
        batch_size = min(512, len(df_all) // 4)  # 动态调整batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Debug: NCF训练 - 批次大小: {batch_size}")
        print(f"Debug: NCF训练 - 总批次数: {len(train_loader)}")
        
        # 训练模型
        print(f"Debug: NCF训练 - 开始训练...")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for users, items, labels in train_loader:
                users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(users, items)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            
            # 计算当前epoch的准确率
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for users, items, labels in train_loader:
                    users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)
                    outputs = self.model(users, items)
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            
            # 记录训练历史
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(accuracy)
            
            print(f"Debug: NCF训练 - Epoch {epoch+1}/{epochs}, 损失: {avg_loss:.6f}, 准确率: {accuracy:.2f}%")
        
        # 训练后评估
        print(f"Debug: NCF训练 - 训练完成，进行模型评估...")
        
        # 损失趋势分析
        if len(self.training_history['losses']) > 1:
            loss_trend = self.training_history['losses'][-1] - self.training_history['losses'][0]
            print(f"Debug: NCF训练 - 损失变化: {self.training_history['losses'][0]:.6f} -> {self.training_history['losses'][-1]:.6f} (变化: {loss_trend:.6f})")
            
            if loss_trend > -0.01:
                print(f"Debug: NCF训练 - 警告: 损失下降不明显，可能需要更多epochs或调整学习率")
        
        # 预测分布分析
        print(f"Debug: NCF训练 - 分析预测分布...")
        sample_users = torch.tensor(list(range(min(100, len(self.user2idx)))), dtype=torch.long).to(self.device)
        sample_items = torch.tensor(list(range(min(100, len(self.item2idx)))), dtype=torch.long).to(self.device)
        
        if len(sample_users) > 0 and len(sample_items) > 0:
            # 创建用户-商品对的网格
            user_grid, item_grid = torch.meshgrid(sample_users, sample_items, indexing='ij')
            flat_users = user_grid.flatten()
            flat_items = item_grid.flatten()
            
            with torch.no_grad():
                sample_outputs = self.model(flat_users, flat_items)
                
            print(f"Debug: NCF训练 - 预测分数统计:")
            print(f"  - 分数范围: {sample_outputs.min().item():.6f} - {sample_outputs.max().item():.6f}")
            print(f"  - 平均分数: {sample_outputs.mean().item():.6f}")
            print(f"  - 分数标准差: {sample_outputs.std().item():.6f}")
            print(f"  - 高分比例(>0.7): {(sample_outputs > 0.7).float().mean().item():.4f}")
            print(f"  - 低分比例(<0.3): {(sample_outputs < 0.3).float().mean().item():.4f}")
        
        self.trained = True
        print(f"Debug: NCF训练完成！")
        print(f"Debug: NCF可推荐用户数: {len(self.user2idx)}")
        print(f"Debug: NCF可推荐商品数: {len(self.item2idx)}")
    
    def recommend(self, user_id_raw, k=10):
        """为用户推荐商品"""
        if not self.trained:
            print(f"Debug: NCF模型未训练")
            return []
            
        user_id = self.user2idx.get(user_id_raw)
        if user_id is None:
            print(f"Debug: 用户 {user_id_raw} 不在NCF训练数据中")
            print(f"Debug: NCF训练数据包含用户数: {len(self.user2idx)}")
            if self.user2idx:
                print(f"Debug: NCF训练数据用户ID范围: {min(self.user2idx.keys())} - {max(self.user2idx.keys())}")
            return []

        user_tensor = torch.tensor([user_id] * len(self.item2idx), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(list(self.item2idx.values()), dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # 检查分数分布
        print(f"Debug: NCF分数范围: {scores.min():.4f} - {scores.max():.4f}")
        print(f"Debug: NCF平均分数: {scores.mean():.4f}")
        
        # 获取推荐商品
        top_items_idx = scores.argsort()[-k:][::-1]
        top_item_ids = [list(self.item2idx.keys())[i] for i in top_items_idx]
        top_scores = [float(scores[i]) for i in top_items_idx]  # 确保分数是float类型
        
        # 如果分数都很低，给出警告而不是归一化
        if max(top_scores) < 0.1:
            print(f"Debug: NCF分数偏低(最高:{max(top_scores):.4f})，可能模型训练不充分")
        
        result = list(zip(top_item_ids, top_scores))
        print(f"Debug: NCF推荐结果数量: {len(result)}")
        print(f"Debug: NCF推荐分数示例: {[f'{score:.4f}' for _, score in result[:3]]}")
        return result

class LSTMRecommender:
    """LSTM序列推荐器"""
    def __init__(self):
        self.model = None
        self.item_to_idx = {}
        self.cat_to_idx = {}
        self.behavior_to_idx = {'pv': 1, 'cart': 2, 'fav': 3, 'buy': 4}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained = False
        self.max_seq_len = 10
        
    def fit(self, df, epochs=3):
        """训练LSTM序列预测模型"""
        print(f"Debug: LSTM训练开始 - 目标epochs: {epochs}")
        
        # 初始化训练记录
        self.training_history = {
            'epochs': [],
            'losses': [],
            'accuracies': []
        }
        
        try:
            # 数据预处理
            df = df.copy()
            print(f"Debug: LSTM训练 - 原始数据行数: {len(df)}")
            
            df = df.sort_values(['user_id', 'timestamp'] if 'timestamp' in df.columns else ['user_id'])
            
            # 构建词汇表
            print(f"Debug: LSTM训练 - 构建词汇表...")
            unique_items = df['item_id'].unique()
            unique_categories = df['category_id'].unique()
            
            self.item_to_idx = {item: idx+1 for idx, item in enumerate(unique_items)}
            self.cat_to_idx = {cat: idx+1 for idx, cat in enumerate(unique_categories)}
            
            print(f"Debug: LSTM训练 - 商品词汇表大小: {len(self.item_to_idx)}")
            print(f"Debug: LSTM训练 - 类别词汇表大小: {len(self.cat_to_idx)}")
            print(f"Debug: LSTM训练 - 行为类型词汇表: {self.behavior_to_idx}")
            
            # 为每个用户构建序列
            print(f"Debug: LSTM训练 - 构建用户行为序列...")
            sequences = []
            all_users = df['user_id'].unique()
            target_users = all_users[:500]  # 限制用户数量以提高训练速度
            
            print(f"Debug: LSTM训练 - 总用户数: {len(all_users)}, 训练用户数: {len(target_users)}")
            
            valid_sequences = 0
            for user_id in target_users:
                user_data = df[df['user_id'] == user_id].copy()
                if len(user_data) < 3:  # 需要至少3条记录
                    continue
                    
                # 映射到索引
                user_data['item_idx'] = user_data['item_id'].map(self.item_to_idx)
                user_data['cat_idx'] = user_data['category_id'].map(self.cat_to_idx)
                user_data['behavior_idx'] = user_data['behavior_type'].map(self.behavior_to_idx)
                
                # 检查映射成功率
                valid_items = user_data['item_idx'].notna().sum()
                valid_cats = user_data['cat_idx'].notna().sum()
                valid_behaviors = user_data['behavior_idx'].notna().sum()
                
                if valid_items < len(user_data) * 0.8 or valid_cats < len(user_data) * 0.8:
                    continue  # 跳过映射成功率低的用户
                
                # 生成序列
                for i in range(2, len(user_data)):
                    seq_items = user_data['item_idx'].iloc[:i].fillna(0).astype(int).tolist()
                    seq_behaviors = user_data['behavior_idx'].iloc[:i].fillna(0).astype(int).tolist()
                    seq_cats = user_data['cat_idx'].iloc[:i].fillna(0).astype(int).tolist()
                    target_cat = user_data['cat_idx'].iloc[i]
                    
                    if pd.isna(target_cat):
                        continue
                    
                    # 时间差特征（简化为位置编码）
                    seq_times = list(range(len(seq_items)))
                    
                    sequences.append({
                        'items': seq_items[-self.max_seq_len:],
                        'behaviors': seq_behaviors[-self.max_seq_len:],
                        'categories': seq_cats[-self.max_seq_len:],
                        'times': seq_times[-self.max_seq_len:],
                        'target': int(target_cat)
                    })
                
                valid_sequences += 1
            
            print(f"Debug: LSTM训练 - 有效用户数: {valid_sequences}")
            print(f"Debug: LSTM训练 - 生成序列数: {len(sequences)}")
            
            if len(sequences) < 10:
                print("Debug: LSTM训练数据不足")
                return False
            
            # 序列长度统计
            seq_lengths = [len(seq['items']) for seq in sequences]
            print(f"Debug: LSTM训练 - 序列长度统计:")
            print(f"  - 平均长度: {np.mean(seq_lengths):.2f}")
            print(f"  - 最大长度: {max(seq_lengths)}")
            print(f"  - 最小长度: {min(seq_lengths)}")
            
            # 目标类别分布
            target_cats = [seq['target'] for seq in sequences]
            target_distribution = pd.Series(target_cats).value_counts()
            print(f"Debug: LSTM训练 - 目标类别分布:")
            print(f"  - 类别数: {len(target_distribution)}")
            print(f"  - 最频繁类别: {target_distribution.index[0]} (出现{target_distribution.iloc[0]}次)")
            print(f"  - 类别分布均匀度: {target_distribution.std()/target_distribution.mean():.3f}")
            
            # 创建模型
            print(f"Debug: LSTM训练 - 创建模型...")
            vocab_size_item = len(self.item_to_idx) + 1
            vocab_size_cat = len(self.cat_to_idx) + 1
            behavior_dim = len(self.behavior_to_idx) + 1
            
            print(f"Debug: LSTM训练 - 模型参数:")
            print(f"  - 商品词汇量: {vocab_size_item}")
            print(f"  - 类别词汇量: {vocab_size_cat}")
            print(f"  - 行为维度: {behavior_dim}")
            print(f"  - 序列长度: {self.max_seq_len}")
            
            self.model = MultiFeatureLSTM(
                item_vocab_size=vocab_size_item,
                behavior_dim=behavior_dim,
                category_vocab_size=vocab_size_cat
            ).to(self.device)
            
            # 模型参数统计
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Debug: LSTM训练 - 模型参数总数: {total_params:,}")
            print(f"Debug: LSTM训练 - 可训练参数数: {trainable_params:,}")
            
            # 保存模型统计信息
            self.model_stats = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'vocab_size_item': vocab_size_item,
                'vocab_size_cat': vocab_size_cat,
                'num_sequences': len(sequences),
                'valid_users': valid_sequences
            }
            
            # 准备训练数据
            print(f"Debug: LSTM训练 - 准备训练数据...")
            train_data = self._prepare_sequences(sequences)
            print(f"Debug: LSTM训练 - 训练批次数: {len(train_data)}")
            
            # 训练模型
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            print(f"Debug: LSTM训练 - 开始训练...")
            
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch in train_data:
                    items, behaviors, categories, times, targets = batch
                    
                    items = items.to(self.device)
                    behaviors = behaviors.to(self.device)
                    categories = categories.to(self.device)
                    times = times.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(items, behaviors, categories, times)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                
                # 计算当前epoch的准确率
                self.model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in train_data:
                        items, behaviors, categories, times, targets = batch
                        
                        items = items.to(self.device)
                        behaviors = behaviors.to(self.device)
                        categories = categories.to(self.device)
                        times = times.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(items, behaviors, categories, times)
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                accuracy = 100 * correct / total if total > 0 else 0
                
                # 记录训练历史
                self.training_history['epochs'].append(epoch + 1)
                self.training_history['losses'].append(avg_loss)
                self.training_history['accuracies'].append(accuracy)
                
                print(f"Debug: LSTM训练 - Epoch {epoch+1}/{epochs}, 损失: {avg_loss:.6f}, 准确率: {accuracy:.2f}%")
                
                # 回到训练模式
                self.model.train()
            
            # 训练后评估
            print(f"Debug: LSTM训练 - 训练完成，进行模型评估...")
            
            # 损失趋势分析
            if len(self.training_history['losses']) > 1:
                loss_trend = self.training_history['losses'][-1] - self.training_history['losses'][0]
                print(f"Debug: LSTM训练 - 损失变化: {self.training_history['losses'][0]:.6f} -> {self.training_history['losses'][-1]:.6f} (变化: {loss_trend:.6f})")
                
                if loss_trend > -0.1:
                    print(f"Debug: LSTM训练 - 警告: 损失下降不明显，可能需要更多epochs或调整学习率")
            
            # 预测分布分析
            print(f"Debug: LSTM训练 - 分析类别预测分布...")
            sample_predictions = []
            
            self.model.eval()
            with torch.no_grad():
                for batch in train_data[:5]:  # 取前5个批次进行分析
                    items, behaviors, categories, times, targets = batch
                    
                    items = items.to(self.device)
                    behaviors = behaviors.to(self.device)
                    categories = categories.to(self.device)
                    times = times.to(self.device)
                    
                    outputs = self.model(items, behaviors, categories, times)
                    probs = torch.softmax(outputs, dim=1)
                    max_probs, predicted_cats = torch.max(probs, 1)
                    
                    sample_predictions.extend(max_probs.cpu().numpy())
            
            if sample_predictions:
                print(f"Debug: LSTM训练 - 预测置信度统计:")
                print(f"  - 置信度范围: {min(sample_predictions):.4f} - {max(sample_predictions):.4f}")
                print(f"  - 平均置信度: {np.mean(sample_predictions):.4f}")
                print(f"  - 高置信度比例(>0.8): {np.mean(np.array(sample_predictions) > 0.8):.4f}")
                print(f"  - 低置信度比例(<0.3): {np.mean(np.array(sample_predictions) < 0.3):.4f}")
            
            self.trained = True
            print(f"Debug: LSTM训练完成！")
            print(f"Debug: LSTM可预测用户需在原始数据中有足够序列")
            return True
            
        except Exception as e:
            print(f"Debug: LSTM训练失败: {str(e)}")
            return False
    
    def _prepare_sequences(self, sequences):
        """准备训练序列"""
        batch_size = 32
        batches = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            
            # Padding
            items_batch = []
            behaviors_batch = []
            categories_batch = []
            times_batch = []
            targets_batch = []
            
            for seq in batch_seqs:
                # Pad sequences to max_seq_len
                items = seq['items'] + [0] * (self.max_seq_len - len(seq['items']))
                behaviors = seq['behaviors'] + [0] * (self.max_seq_len - len(seq['behaviors']))
                categories = seq['categories'] + [0] * (self.max_seq_len - len(seq['categories']))
                times = seq['times'] + [0] * (self.max_seq_len - len(seq['times']))
                
                items_batch.append(items[-self.max_seq_len:])
                behaviors_batch.append(behaviors[-self.max_seq_len:])
                categories_batch.append(categories[-self.max_seq_len:])
                times_batch.append(times[-self.max_seq_len:])
                targets_batch.append(seq['target'])
            
            batch_tensors = (
                torch.tensor(items_batch, dtype=torch.long),
                torch.tensor(behaviors_batch, dtype=torch.long),
                torch.tensor(categories_batch, dtype=torch.long),
                torch.tensor(times_batch, dtype=torch.float),
                torch.tensor(targets_batch, dtype=torch.long)
            )
            batches.append(batch_tensors)
        
        return batches
    
    def recommend_categories(self, user_id, df, k=5):
        """为用户推荐类别"""
        if not self.trained:
            print("Debug: LSTM模型未训练")
            return []
        
        try:
            # 获取用户历史序列
            user_data = df[df['user_id'] == user_id].copy()
            if len(user_data) == 0:
                print(f"Debug: 用户 {user_id} 在原始数据中没有历史数据")
                return []
            
            # 检查用户数据是否足够
            if len(user_data) < 3:
                print(f"Debug: 用户 {user_id} 历史数据不足({len(user_data)}条)，需要至少3条")
                return []
            
            user_data = user_data.sort_values('timestamp' if 'timestamp' in user_data.columns else user_data.columns[0])
            
            # 构建输入序列
            seq_items = [self.item_to_idx.get(item, 0) for item in user_data['item_id'].tail(self.max_seq_len)]
            seq_behaviors = [self.behavior_to_idx.get(behavior, 0) for behavior in user_data['behavior_type'].tail(self.max_seq_len)]
            seq_cats = [self.cat_to_idx.get(cat, 0) for cat in user_data['category_id'].tail(self.max_seq_len)]
            seq_times = list(range(len(seq_items)))
            
            # 检查映射成功率
            valid_items = sum(1 for x in seq_items if x > 0)
            valid_behaviors = sum(1 for x in seq_behaviors if x > 0) 
            valid_cats = sum(1 for x in seq_cats if x > 0)
            
            print(f"Debug: LSTM用户 {user_id} 映射统计 - 商品:{valid_items}/{len(seq_items)}, 行为:{valid_behaviors}/{len(seq_behaviors)}, 类别:{valid_cats}/{len(seq_cats)}")
            
            if valid_items == 0 or valid_cats == 0:
                print(f"Debug: 用户 {user_id} 的商品或类别无法映射到训练词汇表")
                return []
            
            # Padding
            if len(seq_items) < self.max_seq_len:
                pad_len = self.max_seq_len - len(seq_items)
                seq_items = [0] * pad_len + seq_items
                seq_behaviors = [0] * pad_len + seq_behaviors
                seq_cats = [0] * pad_len + seq_cats
                seq_times = [0] * pad_len + seq_times
            
            # 预测
            items_tensor = torch.tensor([seq_items], dtype=torch.long).to(self.device)
            behaviors_tensor = torch.tensor([seq_behaviors], dtype=torch.long).to(self.device)
            cats_tensor = torch.tensor([seq_cats], dtype=torch.long).to(self.device)
            times_tensor = torch.tensor([seq_times], dtype=torch.float).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(items_tensor, behaviors_tensor, cats_tensor, times_tensor)
                probs = torch.softmax(outputs, dim=1)
                topk_probs, topk_ids = torch.topk(probs, k)
            
            # 转换回类别名称
            results = []
            idx_to_cat = {v: k for k, v in self.cat_to_idx.items()}
            
            for prob, cat_id in zip(topk_probs[0], topk_ids[0]):
                cat_id = cat_id.item()
                if cat_id in idx_to_cat:
                    category = idx_to_cat[cat_id]
                    results.append((category, float(prob.item())))
            
            print(f"Debug: LSTM推荐类别数量: {len(results)}")
            print(f"Debug: LSTM推荐类别示例: {[f'{cat}:{prob:.4f}' for cat, prob in results[:3]]}")
            return results
            
        except Exception as e:
            print(f"Debug: LSTM推荐失败: {str(e)}")
            return []
    
    def recommend(self, user_id, df, k=10):
        """为用户推荐商品（基于类别预测）"""
        if not self.trained:
            print("Debug: LSTM模型未训练")
            return []
        
        try:
            # 首先预测用户感兴趣的类别
            category_recommendations = self.recommend_categories(user_id, df, k=min(5, k))
            
            if not category_recommendations:
                print(f"Debug: 用户 {user_id} 无法预测类别")
                return []
            
            # 获取用户历史行为，用于个性化
            user_data = df[df['user_id'] == user_id].copy()
            user_history_items = set(user_data['item_id'].tolist())
            user_preferred_categories = user_data['category_id'].value_counts().to_dict()
            user_behavior_weights = {
                'pv': 0.1, 'cart': 0.3, 'fav': 0.5, 'buy': 1.0
            }
            
            # 计算用户对每个商品的历史偏好分数
            user_item_preference = {}
            for _, row in user_data.iterrows():
                item_id = row['item_id']
                behavior = row['behavior_type']
                weight = user_behavior_weights.get(behavior, 0.1)
                
                if item_id in user_item_preference:
                    user_item_preference[item_id] += weight
                else:
                    user_item_preference[item_id] = weight
            
            # 基于预测的类别推荐商品
            recommendations = []
            
            for category, category_score in category_recommendations:
                # 获取该类别下的商品，排除用户已交互的
                category_items = df[df['category_id'] == category]['item_id'].value_counts()
                
                # 给用户偏好类别更高权重
                category_preference_bonus = user_preferred_categories.get(category, 0) * 0.1
                
                for item_id, popularity in category_items.head(3).items():  # 每个类别取top3
                    if item_id in user_history_items:
                        continue  # 跳过用户已交互的商品
                    
                    # 计算综合分数
                    # 1. 类别预测分数
                    base_score = category_score
                    
                    # 2. 商品热度分数 (归一化)
                    popularity_score = popularity / category_items.max() * 0.3
                    
                    # 3. 类别偏好奖励
                    preference_bonus = category_preference_bonus
                    
                    # 4. 确定性调整因子（基于用户ID，确保一致性）
                    user_hash = hash(str(user_id)) % 1000
                    consistency_factor = 0.9 + (user_hash / 10000)  # 0.9-0.999之间的固定值
                    
                    # 5. 用户历史行为模式匹配度
                    behavior_match = 0.1
                    user_avg_interactions = len(user_data) / user_data['item_id'].nunique() if user_data['item_id'].nunique() > 0 else 1
                    if user_avg_interactions > 2:  # 活跃用户
                        behavior_match = 0.2
                    
                    final_score = (base_score + popularity_score + preference_bonus + behavior_match) * consistency_factor
                    
                    recommendations.append((item_id, float(final_score)))
            
            # 如果推荐数量不足，明确说明原因而不是补充
            if len(recommendations) < k:
                print(f"Debug: LSTM推荐不足({len(recommendations)})，原因分析:")
                if not category_recommendations:
                    print("  - 类别预测失败")
                else:
                    print("  - 预测类别中可推荐商品不足")
                    print(f"  - 预测的类别: {[cat for cat, _ in category_recommendations]}")
                    if user_preferred_categories:
                        print(f"  - 用户历史类别: {list(user_preferred_categories.keys())[:3]}")
            
            # 按分数排序并返回top k
            recommendations.sort(key=lambda x: x[1], reverse=True)
            result = recommendations[:k]
            
            print(f"Debug: LSTM推荐商品数量: {len(result)}")
            if user_preferred_categories:
                print(f"Debug: LSTM用户 {user_id} 偏好类别: {list(user_preferred_categories.keys())[:3]}")
            if result:
                print(f"Debug: LSTM推荐分数范围: {result[0][1]:.4f} - {result[-1][1]:.4f}")
            else:
                print("Debug: LSTM无法为该用户生成推荐")
            return result
            
        except Exception as e:
            print(f"Debug: LSTM商品推荐失败: {str(e)}")
            return []

# ===================== 原有代码继续 =====================

# 设置文件上传大小限制为5GB
@st.cache_resource
def configure_upload_size():
    """配置文件上传大小限制"""
    # Streamlit 默认限制是200MB，我们通过配置将其提升到5GB
    import streamlit.config as stconfig
    try:
        # 设置最大上传文件大小为5120MB (5GB)
        os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '5120'
        return True
    except Exception:
        return False

# 调用配置函数
configure_upload_size()

class RecommendationDashboard:
    """推荐系统可视化界面类"""
    
    def __init__(self):
        self.data = None
        self.user_features = None
        self.recommendations = None
        # 初始化推荐器
        self.cf_recommender = CollaborativeFilteringRecommender()
        self.ncf_recommender = NCFRecommender()
        self.lstm_recommender = LSTMRecommender()
        
        # 初始化session state
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        
        if 'trained_ncf_recommender' not in st.session_state:
            st.session_state.trained_ncf_recommender = None
        
        if 'trained_lstm_recommender' not in st.session_state:
            st.session_state.trained_lstm_recommender = None
    
    @st.cache_data
    def load_data(_self, file_path):
        """加载数据"""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            return None
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🛒 电商推荐系统")
        st.sidebar.markdown("---")
        
        # 数据加载
        st.sidebar.subheader("📁 数据加载")
        
        
        uploaded_file = st.sidebar.file_uploader(
            "选择数据文件", 
            type=['csv'],
            help="请上传包含user_id, item_id, behavior_type, datetime列的CSV文件 (最大5GB)"
        )
        
        if uploaded_file:
            # 显示文件信息
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.sidebar.info(f"📄 文件信息:\n"
                           f"- 文件名: {uploaded_file.name}\n"
                           f"- 文件大小: {file_size_mb:.1f} MB")
            
            # 加载数据，并显示进度
            with st.spinner("正在加载数据..."):
                try:
                    self.data = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"✅ 数据加载成功: {len(self.data):,} 条记录")
                    
                    # 数据预处理 - 时间戳处理
                    if 'timestamp' in self.data.columns and 'timestamp_dt' not in self.data.columns:
                        st.sidebar.info("🕒 检测到原始时间戳，正在转换为北京时间...")
                        # 转换Unix时间戳为datetime，并转换为北京时间（UTC+8）
                        self.data['timestamp_dt'] = pd.to_datetime(self.data['timestamp'], unit='s', errors='coerce')
                        self.data['timestamp_dt'] = self.data['timestamp_dt'] + pd.Timedelta(hours=8)
                        
                        # 添加时间特征
                        self.data['date'] = self.data['timestamp_dt'].dt.date
                        self.data['hour'] = self.data['timestamp_dt'].dt.hour
                        self.data['weekday'] = self.data['timestamp_dt'].dt.day_name()
                        self.data['day_of_week'] = self.data['timestamp_dt'].dt.dayofweek
                        self.data['day_of_month'] = self.data['timestamp_dt'].dt.day
                        self.data['is_weekend'] = self.data['timestamp_dt'].dt.weekday >= 5
                        
                        st.sidebar.success("✅ 时间戳转换完成（已转换为北京时间）")
                    
                    # 显示数据基本信息
                    st.sidebar.write("**数据字段:**")
                    st.sidebar.write(f"- 列数: {len(self.data.columns)}")
                    st.sidebar.write(f"- 字段: {', '.join(self.data.columns.tolist()[:5])}{'...' if len(self.data.columns) > 5 else ''}")
                    
                    # 检查数据格式并提供建议
                    if 'timestamp_dt' in self.data.columns:
                        if 'date' in self.data.columns:
                            st.sidebar.info("📊 检测到预处理数据格式（包含北京时间）")
                        else:
                            st.sidebar.info("📊 检测到时间戳数据，已转换为北京时间")
                    elif 'timestamp' in self.data.columns:
                        st.sidebar.info("📊 检测到原始数据格式")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ 数据加载失败: {str(e)}")
                    st.sidebar.info("请检查文件格式是否正确（CSV格式，包含必要字段）")
        
        # 分析选项
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 分析选项")
        
        analysis_type = st.sidebar.selectbox(
            "选择分析类型",
            ["数据概览", "用户行为分析", "用户画像分析", "推荐算法比较", "个性化推荐"]
        )
        
        return analysis_type
    
    def render_data_overview(self):
        st.header("📊 数据概览与探索性分析")
        
        if self.data is None or self.data.empty:
            st.warning("请先在侧边栏上传数据文件")
            return
        
        df = self.data
        
        # 创建选项卡
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 基础统计", 
            "🎯 单变量分析", 
            "🔍 多变量分析", 
            "👥 用户行为分析",
            "🔄 序列分析"
        ])
        
        with tab1:
            st.subheader("基础统计信息")
            
            # 基础信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总记录数", f"{len(df):,}")
            with col2:
                st.metric("独立用户数", f"{df['user_id'].nunique():,}")
            with col3:
                st.metric("独立商品数", f"{df['item_id'].nunique():,}")
            with col4:
                st.metric("独立类目数", f"{df['category_id'].nunique():,}")
            
            # 数据时间范围
            if 'timestamp_dt' in df.columns:
                st.write(f"**数据时间范围**: {df['timestamp_dt'].min()} 到 {df['timestamp_dt'].max()}")
            elif 'date' in df.columns:
                st.write(f"**数据时间范围**: {df['date'].min()} 到 {df['date'].max()}")
            
            # 数据预览
            st.subheader("数据预览")
            # 修复Arrow序列化问题 - 确保数据类型兼容
            display_df = df.head(10).copy()
            for col in display_df.columns:
                if display_df[col].dtype == 'object':
                    try:
                        display_df[col] = display_df[col].astype(str)
                    except:
                        pass
            st.dataframe(display_df)
            
            # 数据类型
            st.subheader("数据类型")
            st.write(df.dtypes)
            
            # 缺失值统计
            st.subheader("缺失值统计")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.write(missing_data[missing_data > 0])
            else:
                st.success("数据中没有缺失值")
        
        with tab2:
            st.subheader("🎯 单变量分析")
            
            # 行为类型分布
            st.subheader("行为类型分布")
            behavior_counts = df['behavior_type'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**行为类型统计:**")
                st.write(behavior_counts)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=behavior_counts.index, y=behavior_counts.values, palette='viridis', ax=ax)
                ax.set_title('Distribution of Behavior Types')
                ax.set_xlabel('Behavior Type')
                ax.set_ylabel('Count')
                st.pyplot(fig)
            
            # 时间序列分析
            if 'date' in df.columns:
                st.subheader("时间序列分析")
                
                # 按天统计
                st.write("**每日用户行为总量**")
                daily_behavior_counts = df.groupby('date')['user_id'].count()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                daily_behavior_counts.plot(kind='line', marker='o', ax=ax)
                ax.set_title('Total User Behaviors per Day')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Behaviors')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 按小时统计
                if 'hour' in df.columns:
                    st.write("**每小时用户行为总量**")
                    hourly_behavior_counts = df.groupby('hour')['user_id'].count()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    hourly_behavior_counts.plot(kind='bar', color='skyblue', ax=ax)
                    ax.set_title('Total User Behaviors per Hour of Day')
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Number of Behaviors')
                    ax.tick_params(axis='x', rotation=0)
                    ax.grid(axis='y')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 按星期几统计
                if 'weekday' in df.columns:
                    st.write("**每周各天用户行为总量**")
                    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekday_behavior_counts = df.groupby('weekday')['user_id'].count()
                    if all(day in weekday_behavior_counts.index for day in weekday_order):
                        weekday_behavior_counts = weekday_behavior_counts.reindex(weekday_order)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    weekday_behavior_counts.plot(kind='bar', color='lightcoral', ax=ax)
                    ax.set_title('Total User Behaviors per Day of Week')
                    ax.set_xlabel('Day of Week')
                    ax.set_ylabel('Number of Behaviors')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y')
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab3:
            st.subheader("🔍 多变量分析与热门分析")
            
            top_n = st.slider("显示Top N项目", min_value=5, max_value=20, value=10)
            
            # Top N 商品 (基于PV行为)
            st.subheader("热门商品分析")
            pv_df = df[df['behavior_type'] == 'pv']
            
            if not pv_df.empty:
                top_items_pv = pv_df['item_id'].value_counts().head(top_n)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Top {top_n} 最受关注商品 (PV):**")
                    st.write(top_items_pv)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x=top_items_pv.index, y=top_items_pv.values, palette='coolwarm', ax=ax)
                    ax.set_title(f'Top {top_n} Most Viewed Items (PV)')
                    ax.set_xlabel('Item ID')
                    ax.set_ylabel('Number of Page Views (PV)')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Top N 商品类目 (基于PV行为)
                st.write("**热门商品类目**")
                top_categories_pv = pv_df['category_id'].value_counts().head(top_n)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Top {top_n} 最受关注类目 (PV):**")
                    st.write(top_categories_pv)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x=top_categories_pv.index, y=top_categories_pv.values, palette='autumn', ax=ax)
                    ax.set_title(f'Top {top_n} Most Viewed Categories (PV)')
                    ax.set_xlabel('Category ID')
                    ax.set_ylabel('Number of Page Views (PV)')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Top N 购买的商品
            buy_df = df[df['behavior_type'] == 'buy']
            if not buy_df.empty:
                st.subheader("购买行为分析")
                top_items_buy = buy_df['item_id'].value_counts().head(top_n)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Top {top_n} 最多购买商品:**")
                    st.write(top_items_buy)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x=top_items_buy.index, y=top_items_buy.values, palette='winter', ax=ax)
                    ax.set_title(f'Top {top_n} Most Purchased Items')
                    ax.set_xlabel('Item ID')
                    ax.set_ylabel('Number of Purchases')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("数据中没有购买行为，无法显示购买商品分析")
            
            # 不同行为类型的时间分布
            if 'date' in df.columns:
                st.subheader("行为类型时间分布")
                
                # 按日期和行为类型分组
                behaviors_by_date_type = df.groupby(['date', 'behavior_type'])['user_id'].count().unstack('behavior_type').fillna(0)
                
                fig, ax = plt.subplots(figsize=(14, 7))
                behaviors_by_date_type.plot(kind='line', marker='.', ax=ax)
                ax.set_title('User Behaviors per Day by Type')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Behaviors')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(title='Behavior Type')
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                
                if 'hour' in df.columns:
                    # 按小时和行为类型分组
                    behaviors_by_hour_type = df.groupby(['hour', 'behavior_type'])['user_id'].count().unstack('behavior_type').fillna(0)
                    
                    fig, ax = plt.subplots(figsize=(14, 7))
                    behaviors_by_hour_type.plot(kind='line', marker='.', ax=ax)
                    ax.set_title('User Behaviors per Hour by Type')
                    ax.set_xlabel('Hour of Day')
                    ax.set_ylabel('Number of Behaviors')
                    ax.legend(title='Behavior Type')
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab4:
            st.subheader("👥 用户行为分析")
            
            # 用户平均行为次数
            user_behavior_counts = df.groupby('user_id')['behavior_type'].count()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("用户平均行为次数", f"{user_behavior_counts.mean():.2f}")
            with col2:
                st.metric("用户行为次数中位数", f"{user_behavior_counts.median():.2f}")
            with col3:
                st.metric("最活跃用户行为次数", f"{user_behavior_counts.max()}")
            
            # 用户行为分布
            st.subheader("用户行为次数分布")
            fig, ax = plt.subplots(figsize=(10, 6))
            # 使用matplotlib的hist而不是seaborn的histplot来避免兼容性问题
            ax.hist(user_behavior_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Distribution of Number of Behaviors per User')
            ax.set_xlabel('Number of Behaviors')
            ax.set_ylabel('Number of Users')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 活跃用户
            st.subheader("最活跃用户")
            top_n_users = st.slider("显示Top N用户", min_value=5, max_value=20, value=10, key="top_users")
            top_active_users = user_behavior_counts.sort_values(ascending=False).head(top_n_users)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Top {top_n_users} 最活跃用户:**")
                st.write(top_active_users)
            
            with col2:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x=top_active_users.index, y=top_active_users.values, palette="crest", ax=ax)
                ax.set_title(f"Top {top_n_users} Most Active Users")
                ax.set_xlabel("User ID")
                ax.set_ylabel("Total Behaviors")
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # 转化率分析
            st.subheader("转化率分析")
            total_pv = behavior_counts.get('pv', 0)
            total_buy = behavior_counts.get('buy', 0)
            
            if total_pv > 0:
                pv_to_buy_ratio = (total_buy / total_pv) * 100
                st.metric("全局 PV 到 Buy 转化率", f"{pv_to_buy_ratio:.2f}%")
            else:
                st.info("数据中没有PV行为，无法计算转化率")
        
        with tab5:
            st.subheader("🔄 用户行为序列分析")
            
            # 检查是否有必要的时间戳列
            if 'timestamp_dt' not in df.columns and 'timestamp' not in df.columns:
                st.warning("数据中缺少时间戳信息，无法进行序列分析")
                return
            
            with st.spinner("构建用户行为序列..."):
                try:
                    # 确保数据按照用户ID和时间戳排序
                    if 'timestamp_dt' in df.columns:
                        df_sorted = df.sort_values(by=['user_id', 'timestamp_dt'], ascending=True)
                    else:
                        df_sorted = df.sort_values(by=['user_id', 'timestamp'], ascending=True)
                    
                    # 为每个用户构建行为序列
                    user_sequences = df_sorted.groupby('user_id').agg(
                        item_sequence=('item_id', list),
                        behavior_sequence=('behavior_type', list),
                        category_sequence=('category_id', list)
                    ).reset_index()
                    
                    # 计算序列长度
                    user_sequences['sequence_length'] = user_sequences['item_sequence'].apply(len)
                    
                    st.success(f"成功为 {len(user_sequences):,} 个用户构建了行为序列")
                    
                    # 序列长度分析
                    st.subheader("序列长度分析")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("平均序列长度", f"{user_sequences['sequence_length'].mean():.2f}")
                    with col2:
                        st.metric("序列长度中位数", f"{user_sequences['sequence_length'].median():.2f}")
                    with col3:
                        st.metric("最长序列", f"{user_sequences['sequence_length'].max()}")
                    
                    # 序列长度分布
                    fig, ax = plt.subplots(figsize=(12, 7))
                    ax.hist(user_sequences['sequence_length'], bins=100, alpha=0.7, color='lightblue', edgecolor='black')
                    ax.set_title('Distribution of User Sequence Lengths')
                    ax.set_xlabel('Sequence Length (Number of Actions per User)')
                    ax.set_ylabel('Number of Users')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # 购买行为分析
                    st.subheader("购买行为序列分析")
                    
                    def has_purchase(behavior_list):
                        return 'buy' in behavior_list
                    
                    user_sequences['has_purchase'] = user_sequences['behavior_sequence'].apply(has_purchase)
                    purchase_user_count = user_sequences['has_purchase'].sum()
                    total_users_in_sequences = len(user_sequences)
                    purchase_percentage = (purchase_user_count / total_users_in_sequences) * 100 if total_users_in_sequences > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("有购买行为的用户", f"{purchase_user_count:,}")
                    with col2:
                        st.metric("购买用户占比", f"{purchase_percentage:.2f}%")
                    
                    # 购买用户 vs 未购买用户的序列长度对比
                    if purchase_user_count > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        # 使用matplotlib创建箱线图
                        purchase_lengths = user_sequences[user_sequences['has_purchase']]['sequence_length']
                        no_purchase_lengths = user_sequences[~user_sequences['has_purchase']]['sequence_length']
                        
                        ax.boxplot([no_purchase_lengths, purchase_lengths], labels=['No Purchase', 'Has Purchase'])
                        ax.set_title('Sequence Length by Purchase Behavior')
                        ax.set_ylabel('Sequence Length')
                        ax.set_yscale('log')
                        st.pyplot(fig)
                    
                    # 行为类型统计
                    st.subheader("用户行为类型统计")
                    behavior_types = ['pv', 'cart', 'fav', 'buy']
                    
                    for b_type in behavior_types:
                        if b_type in df['behavior_type'].values:
                            user_sequences[f'{b_type}_count'] = user_sequences['behavior_sequence'].apply(lambda x: x.count(b_type))
                    
                    # 显示统计信息
                    stats_cols = [col for col in user_sequences.columns if col.endswith('_count')]
                    if stats_cols:
                        st.write("**各行为类型统计描述:**")
                        st.write(user_sequences[stats_cols].describe())
                    
                    # 用户兴趣多样性
                    st.subheader("用户兴趣多样性")
                    user_sequences['unique_items_count'] = user_sequences['item_sequence'].apply(lambda x: len(set(x)))
                    user_sequences['unique_categories_count'] = user_sequences['category_sequence'].apply(lambda x: len(set(x)))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("平均关注商品数", f"{user_sequences['unique_items_count'].mean():.2f}")
                    with col2:
                        st.metric("平均关注类目数", f"{user_sequences['unique_categories_count'].mean():.2f}")
                    
                except Exception as e:
                    st.error(f"序列分析过程中出现错误: {str(e)}")
                    st.info("这可能是由于数据量过大或格式问题导致的。建议尝试使用较小的数据样本。")
    
    def render_user_behavior_analysis(self):
        """渲染用户行为分析页面"""
        st.title("👥 用户行为分析")
        
        if self.data is None:
            st.warning("⚠️ 请先在侧边栏上传数据文件")
            return
        
        # 用户活跃度分析
        st.subheader("📈 用户活跃度分析")
        
        # 动态检查可用的时间列
        time_columns = ['timestamp_dt', 'date', 'datetime', 'timestamp']
        available_time_column = None
        for col in time_columns:
            if col in self.data.columns:
                available_time_column = col
                break
        
        # 构建聚合字典
        agg_dict = {
            'behavior_type': 'count',
            'item_id': 'nunique'
        }
        
        # 如果有时间列，添加时间相关的聚合
        if available_time_column:
            agg_dict[available_time_column] = ['min', 'max']
            column_names = ['总行为数', '浏览商品数', '首次活跃', '最后活跃']
        else:
            column_names = ['总行为数', '浏览商品数']
        
        user_activity = self.data.groupby('user_id').agg(agg_dict).round(2)
        user_activity.columns = column_names
        
        # 活跃度分布
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                user_activity['总行为数'],
                title="用户活跃度分布",
                labels={'value': '总行为数', 'count': '用户数量'},
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                user_activity['浏览商品数'],
                title="用户浏览商品数分布",
                labels={'value': '浏览商品数', 'count': '用户数量'},
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 用户行为模式分析
        st.subheader("🎯 用户行为模式")
        
        # 计算用户转化率
        user_behavior_analysis = []
        
        for user_id in self.data['user_id'].unique()[:1000]:  # 限制分析用户数量
            user_data = self.data[self.data['user_id'] == user_id]
            behavior_counts = user_data['behavior_type'].value_counts()
            
            pv_count = behavior_counts.get('pv', 0)
            cart_count = behavior_counts.get('cart', 0)
            fav_count = behavior_counts.get('fav', 0)
            buy_count = behavior_counts.get('buy', 0)
            
            user_behavior_analysis.append({
                'user_id': user_id,
                'pv_count': pv_count,
                'cart_count': cart_count,
                'fav_count': fav_count,
                'buy_count': buy_count,
                'pv_to_cart_rate': cart_count / pv_count if pv_count > 0 else 0,
                'pv_to_buy_rate': buy_count / pv_count if pv_count > 0 else 0,
                'cart_to_buy_rate': buy_count / cart_count if cart_count > 0 else 0
            })
        
        behavior_df = pd.DataFrame(user_behavior_analysis)
        
        # 转化率分布
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.histogram(
                behavior_df['pv_to_cart_rate'],
                title="浏览到加购转化率分布",
                labels={'value': '转化率', 'count': '用户数量'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                behavior_df['pv_to_buy_rate'],
                title="浏览到购买转化率分布",
                labels={'value': '转化率', 'count': '用户数量'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(
                behavior_df['cart_to_buy_rate'],
                title="加购到购买转化率分布",
                labels={'value': '转化率', 'count': '用户数量'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 用户行为统计表
        st.subheader("📊 用户行为统计")
        
        summary_stats = behavior_df[['pv_to_cart_rate', 'pv_to_buy_rate', 'cart_to_buy_rate']].describe()
        st.dataframe(summary_stats, use_container_width=True)
    
    def render_user_segmentation(self):
        """用户画像分析页面 - 基于K-Means聚类的用户分群"""
        st.header("👥 用户画像分析")
        
        if self.data is None or self.data.empty:
            st.warning("请先在侧边栏上传数据文件")
            return
        
        df = self.data
        
        # 检查是否有必要的时间戳列
        if 'timestamp_dt' not in df.columns and 'timestamp' not in df.columns:
            st.warning("数据中缺少时间戳信息，无法进行用户画像分析")
            return
        
        st.markdown("通过用户行为序列特征进行聚类分析，识别不同的用户群体类型")
        
        with st.spinner("构建用户行为序列特征..."):
            try:
                # 确保数据按照用户ID和时间戳排序
                if 'timestamp_dt' in df.columns:
                    df_sorted = df.sort_values(by=['user_id', 'timestamp_dt'], ascending=True)
                else:
                    df_sorted = df.sort_values(by=['user_id', 'timestamp'], ascending=True)
                
                # 为每个用户构建行为序列
                user_sequences = df_sorted.groupby('user_id').agg(
                    item_sequence=('item_id', list),
                    behavior_sequence=('behavior_type', list),
                    category_sequence=('category_id', list)
                ).reset_index()
                
                # 计算聚类特征
                user_sequences['sequence_length'] = user_sequences['item_sequence'].apply(len)
                
                # 各种行为类型的计数
                behavior_types = ['pv', 'cart', 'fav', 'buy']
                for b_type in behavior_types:
                    user_sequences[f'{b_type}_count'] = user_sequences['behavior_sequence'].apply(lambda x: x.count(b_type))
                
                # 交互的独立商品数和类目数
                user_sequences['unique_items_count'] = user_sequences['item_sequence'].apply(lambda x: len(set(x)))
                user_sequences['unique_categories_count'] = user_sequences['category_sequence'].apply(lambda x: len(set(x)))
                
                # 购买转化率
                user_sequences['user_pv_to_buy_conversion_rate'] = user_sequences.apply(
                    lambda row: (row['buy_count'] / row['pv_count'] * 100) if row['pv_count'] > 0 else 0, axis=1
                )
                
                st.success(f"✅ 成功构建 {len(user_sequences):,} 个用户的行为特征")
                
                # 创建选项卡
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🔧 特征工程", 
                    "📊 聚类分析", 
                    "🎯 用户画像",
                    "📈 群体对比",
                    "📊 RFM分析"
                ])
                
                with tab1:
                    st.subheader("🔧 用户行为特征工程")
                    
                    # 显示特征统计
                    features_for_clustering = [
                        'sequence_length', 'pv_count', 'cart_count', 'fav_count', 
                        'buy_count', 'unique_items_count', 'unique_categories_count', 
                        'user_pv_to_buy_conversion_rate'
                    ]
                    
                    st.write("**选择的聚类特征:**")
                    for feature in features_for_clustering:
                        st.write(f"- {feature}")
                    
                    # 特征描述性统计
                    st.subheader("特征描述性统计")
                    clustering_data = user_sequences[features_for_clustering].copy()
                    
                    # 处理异常值
                    if clustering_data.isnull().sum().any():
                        st.warning("⚠️ 检测到空值，将用中位数填充")
                        for col in clustering_data.columns[clustering_data.isnull().any()]:
                            clustering_data[col] = clustering_data[col].fillna(clustering_data[col].median())
                    
                    if np.isinf(clustering_data.values).any():
                        st.warning("⚠️ 检测到无穷值，将进行处理")
                        clustering_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                        for col in clustering_data.columns[clustering_data.isnull().any()]:
                            clustering_data[col] = clustering_data[col].fillna(clustering_data[col].median())
                    
                    st.write(clustering_data.describe())
                    
                    # 特征分布可视化
                    st.subheader("特征分布可视化")
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    axes = axes.ravel()
                    
                    for i, feature in enumerate(features_for_clustering):
                        axes[i].hist(clustering_data[feature], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                        axes[i].set_title(f'{feature}')
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('频次')
                        axes[i].set_yscale('log')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab2:
                    st.subheader("📊 K-Means 聚类分析")
                    
                    # 特征标准化
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(clustering_data)
                    st.success("✅ 特征标准化完成")
                    
                    # 肘部法则确定最优K值
                    st.subheader("肘部法则确定最优聚类数")
                    
                    with st.spinner("计算不同K值的惯性..."):
                        possible_k_values = range(2, 11)
                        inertia_values = []
                        
                        for k in possible_k_values:
                            kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
                            kmeans_temp.fit(scaled_features)
                            inertia_values.append(kmeans_temp.inertia_)
                    
                    # 绘制肘部法则图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(possible_k_values, inertia_values, marker='o', linestyle='-', linewidth=2, markersize=8)
                    ax.set_title('肘部法则确定最优K值', fontsize=14)
                    ax.set_xlabel('聚类数量 (K)')
                    ax.set_ylabel('惯性值 (Inertia)')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_xticks(possible_k_values)
                    
                    # 添加数值标签
                    for i, (k, inertia) in enumerate(zip(possible_k_values, inertia_values)):
                        ax.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=9)
                    
                    st.pyplot(fig)
                    
                    # 让用户选择K值
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**各K值对应的惯性值:**")
                        for k, inertia in zip(possible_k_values, inertia_values):
                            st.write(f"K={k}: {inertia:.2f}")
                    
                    with col2:
                        chosen_k = st.selectbox(
                            "根据肘部法则图选择最优K值:",
                            options=list(possible_k_values),
                            index=3,  # 默认选择K=5
                            help="通常选择惯性值下降趋势明显放缓的拐点"
                        )
                    
                    # 执行聚类
                    if st.button("🚀 执行K-Means聚类", type="primary"):
                        with st.spinner(f"执行K={chosen_k}聚类分析..."):
                            kmeans = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, random_state=42)
                            cluster_labels = kmeans.fit_predict(scaled_features)
                            
                            # 将聚类结果保存到session state
                            st.session_state.user_sequences = user_sequences.copy()
                            st.session_state.user_sequences['cluster'] = cluster_labels
                            st.session_state.chosen_k = chosen_k
                            st.session_state.clustering_data = clustering_data
                            st.session_state.features_for_clustering = features_for_clustering
                            
                            st.success(f"✅ 聚类完成！成功将用户分为 {chosen_k} 个群体")
                
                with tab3:
                    st.subheader("🎯 用户群体画像")
                    
                    if 'user_sequences' not in st.session_state:
                        st.info("请先在 '聚类分析' 标签页执行聚类分析")
                        return
                    
                    user_sequences_with_clusters = st.session_state.user_sequences
                    chosen_k = st.session_state.chosen_k
                    features_for_clustering = st.session_state.features_for_clustering
                    
                    # 各群体用户数量
                    cluster_counts = user_sequences_with_clusters['cluster'].value_counts().sort_index()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**各群体用户数量:**")
                        for cluster_id, count in cluster_counts.items():
                            percentage = (count / len(user_sequences_with_clusters)) * 100
                            st.write(f"群体 {cluster_id}: {count:,} 用户 ({percentage:.1f}%)")
                    
                    with col2:
                        # 群体分布饼图
                        fig, ax = plt.subplots(figsize=(8, 8))
                        colors = plt.cm.Set3(np.linspace(0, 1, chosen_k))
                        wedges, texts, autotexts = ax.pie(cluster_counts.values, 
                                                         labels=[f'群体 {i}' for i in cluster_counts.index],
                                                         autopct='%1.1f%%',
                                                         colors=colors,
                                                         startangle=90)
                        ax.set_title(f'用户群体分布 (K={chosen_k})', fontsize=14)
                        st.pyplot(fig)
                    
                    # 群体特征画像
                    st.subheader("群体特征画像对比")
                    cluster_profiles = user_sequences_with_clusters.groupby('cluster')[features_for_clustering].mean()
                    
                    # 显示数值表格
                    st.write("**各群体特征均值:**")
                    st.dataframe(cluster_profiles.round(2))
                    
                    # 可视化群体画像
                    profile_plot_data = cluster_profiles.reset_index().melt(
                        id_vars='cluster', var_name='feature', value_name='mean_value'
                    )
                    
                    # 创建子图
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    axes = axes.ravel()
                    
                    for i, feature in enumerate(features_for_clustering):
                        feature_data = profile_plot_data[profile_plot_data['feature'] == feature]
                        
                        bars = axes[i].bar(feature_data['cluster'], feature_data['mean_value'], 
                                          color=plt.cm.viridis(np.linspace(0, 1, chosen_k)))
                        axes[i].set_title(f'{feature}', fontsize=12)
                        axes[i].set_xlabel('群体 ID')
                        axes[i].set_ylabel('平均值')
                        axes[i].grid(True, alpha=0.3)
                        
                        # 添加数值标签
                        for bar, value in zip(bars, feature_data['mean_value']):
                            height = bar.get_height()
                            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.suptitle(f'各群体特征画像对比 (K={chosen_k})', fontsize=16, y=1.02)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab4:
                    st.subheader("📈 用户群体深度对比")
                    
                    if 'user_sequences' not in st.session_state:
                        st.info("请先在 '聚类分析' 标签页执行聚类分析")
                        return
                    
                    user_sequences_with_clusters = st.session_state.user_sequences
                    
                    # 选择要对比的群体
                    cluster_ids = sorted(user_sequences_with_clusters['cluster'].unique())
                    selected_clusters = st.multiselect(
                        "选择要对比的用户群体:",
                        options=cluster_ids,
                        default=cluster_ids[:3] if len(cluster_ids) >= 3 else cluster_ids,
                        help="可以选择多个群体进行对比分析"
                    )
                    
                    if len(selected_clusters) < 2:
                        st.warning("请至少选择2个群体进行对比")
                        return
                    
                    # 购买行为对比
                    st.subheader("购买行为对比")
                    
                    def has_purchase(behavior_list):
                        return 'buy' in behavior_list
                    
                    purchase_stats = []
                    for cluster_id in selected_clusters:
                        cluster_data = user_sequences_with_clusters[user_sequences_with_clusters['cluster'] == cluster_id]
                        has_purchase_count = cluster_data['behavior_sequence'].apply(has_purchase).sum()
                        total_users = len(cluster_data)
                        purchase_rate = (has_purchase_count / total_users * 100) if total_users > 0 else 0
                        
                        purchase_stats.append({
                            'cluster': f'群体 {cluster_id}',
                            'total_users': total_users,
                            'buyers': has_purchase_count,
                            'purchase_rate': purchase_rate
                        })
                    
                    purchase_df = pd.DataFrame(purchase_stats)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**购买行为统计:**")
                        st.dataframe(purchase_df)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(purchase_df['cluster'], purchase_df['purchase_rate'], 
                                     color=plt.cm.viridis(np.linspace(0, 1, len(selected_clusters))))
                        ax.set_title('各群体购买转化率对比')
                        ax.set_ylabel('购买转化率 (%)')
                        ax.set_xlabel('用户群体')
                        
                        # 添加数值标签
                        for bar, rate in zip(bars, purchase_df['purchase_rate']):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{rate:.1f}%', ha='center', va='bottom')
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # 行为序列长度对比
                    st.subheader("行为活跃度对比")
                    
                    sequence_length_data = []
                    for cluster_id in selected_clusters:
                        cluster_data = user_sequences_with_clusters[user_sequences_with_clusters['cluster'] == cluster_id]
                        sequence_length_data.extend([(f'群体 {cluster_id}', length) 
                                                   for length in cluster_data['sequence_length']])
                    
                    sequence_df = pd.DataFrame(sequence_length_data, columns=['cluster', 'sequence_length'])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 创建箱线图
                    cluster_names = [f'群体 {cid}' for cid in selected_clusters]
                    sequence_data_by_cluster = [sequence_df[sequence_df['cluster'] == name]['sequence_length'].values 
                                              for name in cluster_names]
                    
                    box_plot = ax.boxplot(sequence_data_by_cluster, labels=cluster_names, patch_artist=True)
                    
                    # 设置颜色
                    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_clusters)))
                    for patch, color in zip(box_plot['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_title('各群体用户行为序列长度分布')
                    ax.set_ylabel('序列长度（行为次数）')
                    ax.set_xlabel('用户群体')
                    ax.set_yscale('log')
                    ax.grid(True, alpha=0.3)
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 用户画像解读
                st.markdown("---")
                st.subheader("📋 用户画像解读建议")
                st.markdown("""
                **基于聚类结果，可以从以下角度解读用户群体:**
                
                1. **高价值用户群** - 购买频次高、转化率高、活跃度高
                   - 特征：buy_count高、user_pv_to_buy_conversion_rate高、sequence_length较高
                   - 策略：VIP服务、忠诚度计划、高端商品推荐
                
                2. **潜力用户群** - 活跃度高但购买较少
                   - 特征：sequence_length高、pv_count高，但buy_count低
                   - 策略：精准推荐、优惠促销、购买引导
                
                3. **浏览型用户** - 浏览多但很少购买
                   - 特征：pv_count高、cart_count或fav_count一般，buy_count很低
                   - 策略：内容优化、兴趣引导、信任建设
                
                4. **低频用户** - 各项指标都较低
                   - 特征：所有计数指标都偏低
                   - 策略：激活营销、新用户引导、基础推荐
                
                5. **目标明确用户** - 浏览少但转化率高
                   - 特征：pv_count相对较低但buy_count不错
                   - 策略：精准匹配、快速响应、简化流程
                """)
                
                with tab5:
                    st.subheader("📊 RFM分析")
                    st.markdown("基于最近性(Recency)、频率(Frequency)、货币价值(Monetary)进行用户价值分析")
                    
                    # 检查数据是否有时间戳
                    if 'timestamp_dt' not in df.columns and 'date' not in df.columns:
                        st.warning("需要时间信息来计算RFM指标，请确保数据包含时间戳")
                        return
                    
                    with st.spinner("计算RFM指标..."):
                        # 计算RFM指标
                        try:
                            # 确定当前日期
                            if 'timestamp_dt' in df.columns:
                                current_date = pd.to_datetime(df['timestamp_dt']).max()
                                date_column = 'timestamp_dt'
                            elif 'date' in df.columns:
                                current_date = pd.to_datetime(df['date']).max()
                                date_column = 'date'
                            else:
                                st.error("数据中没有找到有效的时间列")
                                return
                        except Exception as e:
                            st.error(f"日期解析错误: {str(e)}")
                            return
                        
                        rfm_data = []
                        unique_users = df['user_id'].unique()
                        
                        # 为了演示，我们取前10000个用户（如果用户数过多）
                        if len(unique_users) > 10000:
                            st.info(f"用户数量较多({len(unique_users):,})，将分析前10,000个用户")
                            unique_users = unique_users[:10000]
                        
                        # 添加进度条
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        
                        for i, user_id in enumerate(unique_users):
                            # 更新进度
                            if i % 1000 == 0:
                                progress = (i + 1) / len(unique_users)
                                progress_bar.progress(progress)
                                progress_text.text(f"处理进度: {i+1:,}/{len(unique_users):,} 用户")
                            
                            try:
                                user_data = df[df['user_id'] == user_id]
                                
                                # R - Recency: 最近一次交互距今天数
                                if date_column == 'timestamp_dt':
                                    last_interaction = pd.to_datetime(user_data['timestamp_dt']).max()
                                else:
                                    last_interaction = pd.to_datetime(user_data['date']).max()
                                
                                # 计算天数差
                                recency = (current_date - last_interaction).days
                                
                                # 确保recency是有效数值
                                if pd.isna(recency) or recency < 0:
                                    recency = 999  # 给一个默认的大值
                                
                                # F - Frequency: 交互频率（总行为次数）
                                frequency = len(user_data)
                                
                                # M - Monetary: 货币价值（这里用购买次数代替，因为没有金额数据）
                                monetary = len(user_data[user_data['behavior_type'] == 'buy']) if 'buy' in user_data['behavior_type'].values else 0
                                
                                # 计算额外的行为指标
                                pv_count = len(user_data[user_data['behavior_type'] == 'pv'])
                                cart_count = len(user_data[user_data['behavior_type'] == 'cart'])
                                fav_count = len(user_data[user_data['behavior_type'] == 'fav'])
                                
                                # RFM分群规则（调整阈值使其更合理）
                                if recency <= 3 and frequency >= 10 and monetary >= 2:
                                    segment = "冠军用户"
                                elif recency <= 7 and frequency >= 5 and monetary >= 1:
                                    segment = "忠诚用户"
                                elif recency <= 3 and frequency < 5:
                                    segment = "新用户"
                                elif recency > 7 and frequency >= 5:
                                    segment = "流失风险用户"
                                elif monetary == 0 and frequency >= 3:
                                    segment = "潜在用户"
                                else:
                                    segment = "一般用户"
                                
                                rfm_data.append({
                                    'user_id': user_id,
                                    'recency': int(recency),  # 确保是整数
                                    'frequency': int(frequency),
                                    'monetary': int(monetary),
                                    'pv_count': int(pv_count),
                                    'cart_count': int(cart_count),
                                    'fav_count': int(fav_count),
                                    'segment': segment
                                })
                                
                            except Exception as user_error:
                                # 如果某个用户处理失败，跳过该用户
                                st.warning(f"跳过用户 {user_id}: {str(user_error)}")
                                continue
                        
                        # 清除进度条
                        progress_bar.empty()
                        progress_text.empty()
                        
                        if not rfm_data:
                            st.error("没有成功处理任何用户数据，请检查数据格式")
                            return
                        
                        rfm_df = pd.DataFrame(rfm_data)
                        
                        # 数据验证
                        if len(rfm_df) == 0:
                            st.error("RFM计算结果为空，请检查数据")
                            return
                        
                        st.success(f"✅ 成功计算 {len(rfm_df):,} 个用户的RFM指标")
                    
                    # RFM概览
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("平均最近性", f"{rfm_df['recency'].mean():.1f} 天")
                    with col2:
                        st.metric("平均频率", f"{rfm_df['frequency'].mean():.1f} 次")
                    with col3:
                        st.metric("平均购买次数", f"{rfm_df['monetary'].mean():.1f} 次")
                    
                    # 数据类型确保
                    rfm_df['recency'] = pd.to_numeric(rfm_df['recency'], errors='coerce').fillna(999).astype(int)
                    rfm_df['frequency'] = pd.to_numeric(rfm_df['frequency'], errors='coerce').fillna(0).astype(int)
                    rfm_df['monetary'] = pd.to_numeric(rfm_df['monetary'], errors='coerce').fillna(0).astype(int)
                    
                    # RFM分群分布
                    st.subheader("RFM用户分群分布")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 分群分布饼图
                        segment_counts = rfm_df['segment'].value_counts()
                        fig = px.pie(
                            values=segment_counts.values,
                            names=segment_counts.index,
                            title="RFM用户分群分布",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 分群数量柱状图
                        fig = px.bar(
                            x=segment_counts.index,
                            y=segment_counts.values,
                            title="各分群用户数量",
                            labels={'x': '用户分群', 'y': '用户数量'},
                            color=segment_counts.values,
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # RFM 3D散点图
                    st.subheader("RFM 3D散点图")
                    
                    try:
                        # 确保数据类型正确
                        plot_data = rfm_df.copy()
                        plot_data = plot_data.dropna(subset=['recency', 'frequency', 'monetary'])
                        
                        if len(plot_data) == 0:
                            st.error("没有有效的RFM数据用于绘制3D图")
                            return
                        
                        # 创建3D散点图
                        fig = px.scatter_3d(
                            plot_data,
                            x='recency',
                            y='frequency',
                            z='monetary',
                            color='segment',
                            title="RFM三维分布",
                            labels={
                                'recency': '最近性 (天)',
                                'frequency': '频率 (次)',
                                'monetary': '购买次数'
                            },
                            hover_data=['user_id', 'pv_count', 'cart_count', 'fav_count'],
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        
                        # 优化3D图的显示
                        fig.update_traces(
                            marker=dict(size=5, opacity=0.7),
                            selector=dict(mode='markers')
                        )
                        
                        fig.update_layout(
                            scene=dict(
                                xaxis_title="最近性 (天) - 越小越好",
                                yaxis_title="频率 (次) - 越大越好", 
                                zaxis_title="购买次数 - 越大越好",
                                camera=dict(
                                    eye=dict(x=1.5, y=1.5, z=1.5)
                                )
                            ),
                            width=800,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as plot_error:
                        st.error(f"3D散点图绘制失败: {str(plot_error)}")
                        st.info("尝试显示简化的2D图表")
                        
                        # 备用2D图表
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.scatter(rfm_df, x='recency', y='frequency', color='segment',
                                           title="最近性 vs 频率")
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig = px.scatter(rfm_df, x='frequency', y='monetary', color='segment',
                                           title="频率 vs 购买次数")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # RFM分群特征对比
                    st.subheader("RFM分群特征对比")
                    
                    try:
                        segment_summary = rfm_df.groupby('segment').agg({
                            'recency': 'mean',
                            'frequency': 'mean',
                            'monetary': 'mean',
                            'pv_count': 'mean',
                            'cart_count': 'mean',
                            'fav_count': 'mean'
                        }).round(2)
                        
                        segment_summary.columns = ['平均最近性(天)', '平均频率', '平均购买次数', '平均浏览次数', '平均加购次数', '平均收藏次数']
                        st.dataframe(segment_summary, use_container_width=True)
                        
                    except Exception as summary_error:
                        st.error(f"分群特征对比计算失败: {str(summary_error)}")
                        st.write("显示原始数据预览:")
                        # 修复数据显示问题
                        preview_df = rfm_df.head().copy()
                        for col in preview_df.columns:
                            if preview_df[col].dtype == 'object':
                                try:
                                    preview_df[col] = preview_df[col].astype(str)
                                except:
                                    pass
                        st.dataframe(preview_df, use_container_width=True)
                    
                    # 分群详情
                    st.subheader("分群详情分析")
                    
                    try:
                        selected_segment = st.selectbox(
                            "选择要查看的用户分群",
                            options=rfm_df['segment'].unique(),
                            key="rfm_segment_select"
                        )
                        
                        segment_users = rfm_df[rfm_df['segment'] == selected_segment]
                        
                        if len(segment_users) == 0:
                            st.warning(f"分群 '{selected_segment}' 中没有用户")
                            return
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{selected_segment}** 包含 {len(segment_users):,} 个用户")
                            st.write(f"占总用户的 {(len(segment_users)/len(rfm_df)*100):.1f}%")
                            
                            # 该分群的统计信息
                            st.write("**分群特征:**")
                            st.write(f"- 平均最近性: {segment_users['recency'].mean():.1f} 天")
                            st.write(f"- 平均频率: {segment_users['frequency'].mean():.1f} 次")
                            st.write(f"- 平均购买: {segment_users['monetary'].mean():.1f} 次")
                        
                        with col2:
                            try:
                                # 该分群的RFM分布
                                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                                
                                # 检查数据是否为空
                                if len(segment_users) > 0:
                                    axes[0].hist(segment_users['recency'], bins=min(20, len(segment_users)), 
                                                alpha=0.7, color='skyblue', edgecolor='black')
                                    axes[0].set_title('最近性分布')
                                    axes[0].set_xlabel('天数')
                                    axes[0].set_ylabel('用户数')
                                    
                                    axes[1].hist(segment_users['frequency'], bins=min(20, len(segment_users)), 
                                                alpha=0.7, color='lightgreen', edgecolor='black')
                                    axes[1].set_title('频率分布')
                                    axes[1].set_xlabel('交互次数')
                                    axes[1].set_ylabel('用户数')
                                    
                                    axes[2].hist(segment_users['monetary'], bins=min(20, len(segment_users)), 
                                                alpha=0.7, color='lightcoral', edgecolor='black')
                                    axes[2].set_title('购买次数分布')
                                    axes[2].set_xlabel('购买次数')
                                    axes[2].set_ylabel('用户数')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.info("该分群没有足够数据进行分布图绘制")
                                    
                            except Exception as hist_error:
                                st.error(f"分布图绘制失败: {str(hist_error)}")
                        
                        # 显示该分群的用户样本
                        st.write("**用户样本数据:**")
                        display_columns = ['user_id', 'recency', 'frequency', 'monetary', 'pv_count', 'cart_count', 'fav_count']
                        available_columns = [col for col in display_columns if col in segment_users.columns]
                        st.dataframe(segment_users[available_columns].head(20), use_container_width=True)
                        
                    except Exception as detail_error:
                        st.error(f"分群详情分析失败: {str(detail_error)}")
                        st.write("显示基本分群信息:")
                        st.write(rfm_df['segment'].value_counts())
                    
                    # RFM营销建议
                    st.subheader("📈 RFM营销策略建议")
                    
                    strategy_recommendations = {
                        "冠军用户": {
                            "特征": "最近购买、购买频次高、消费金额高",
                            "策略": "VIP专属服务、新品预览、忠诚度奖励、个性化推荐",
                            "重点": "维护关系，提升客单价"
                        },
                        "忠诚用户": {
                            "特征": "购买频次较高，但最近性一般",
                            "策略": "会员权益、定期优惠、生日特权、社群建设",
                            "重点": "增加互动频次，防止流失"
                        },
                        "新用户": {
                            "特征": "最近有交互，但频次和消费较低",
                            "策略": "新用户引导、首购优惠、教育内容、简化流程",
                            "重点": "快速转化，建立习惯"
                        },
                        "流失风险用户": {
                            "特征": "曾经活跃，但最近交互减少",
                            "策略": "召回活动、限时优惠、问卷调研、重新激活",
                            "重点": "及时挽回，找出流失原因"
                        },
                        "潜在用户": {
                            "特征": "有一定活跃度但从未购买",
                            "策略": "购买引导、试用活动、信任建设、降低门槛",
                            "重点": "转化为付费用户"
                        },
                        "一般用户": {
                            "特征": "各项指标都中等",
                            "策略": "分层营销、兴趣探索、个性化内容、逐步培养",
                            "重点": "提升活跃度和价值"
                        }
                    }
                    
                    for segment, info in strategy_recommendations.items():
                        if segment in rfm_df['segment'].unique():
                            with st.expander(f"🎯 {segment} 营销策略"):
                                st.write(f"**用户特征:** {info['特征']}")
                                st.write(f"**营销策略:** {info['策略']}")
                                st.write(f"**重点关注:** {info['重点']}")
                                
                                # 显示该分群的用户数和占比
                                segment_count = len(rfm_df[rfm_df['segment'] == segment])
                                segment_percent = (segment_count / len(rfm_df)) * 100
                                st.write(f"**分群规模:** {segment_count:,} 用户 ({segment_percent:.1f}%)")
            except Exception as e:
                st.error(f"用户画像分析过程中出现错误: {str(e)}")
                st.info("这可能是由于数据量过大或格式问题导致的。建议尝试使用较小的数据样本。")
    
    def render_algorithm_comparison(self):
        """渲染推荐算法比较页面"""
        st.title("🔬 推荐算法比较")
        
        if self.data is None:
            st.warning("⚠️ 请先在侧边栏上传数据文件")
            return
        
        # 模型训练部分
        st.subheader("🎯 模型训练")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if not st.session_state.models_trained:
                if st.button("🚀 开始训练模型", type="primary"):
                    self.train_models()
            else:
                st.success("✅ 模型已训练完成")
                if st.button("🔄 重新训练模型", type="secondary"):
                    # 重置模型状态
                    st.session_state.models_trained = False
                    st.session_state.trained_ncf_recommender = None
                    st.session_state.trained_lstm_recommender = None
                    
                    st.warning("⚠️ 模型状态已重置，请重新训练")
                    st.experimental_rerun()
        
        with col2:
            if st.session_state.models_trained:
                st.info("""
                **已训练的模型:**
                - ✅ NCF深度学习推荐器 (神经协同过滤)
                - ✅ LSTM序列预测器 (基于用户行为序列)
                """)
            else:
                st.info("""
                **待训练的模型:**
                - ⏳ NCF深度学习推荐器  
                - ⏳ LSTM序列预测器
                
                点击"开始训练模型"按钮开始训练
                """)
        
        # 如果模型已训练，显示性能比较
        if st.session_state.models_trained:
            self.render_model_performance_comparison()
        else:
            st.info("请先训练模型以查看性能比较结果")
    
    def render_model_performance_comparison(self):
        """渲染模型性能比较部分"""
        st.subheader("📊 模型性能比较")
        
        # 创建选项卡
        tab1, tab2, tab3 = st.tabs(["📈 训练过程", "🎯 雷达图比较", "📋 详细统计"])
        
        with tab1:
            st.subheader("训练过程可视化")
            
            # 获取训练历史数据
            ncf_history = None
            lstm_history = None
            
            if (st.session_state.trained_ncf_recommender and 
                hasattr(st.session_state.trained_ncf_recommender, 'training_history')):
                ncf_history = st.session_state.trained_ncf_recommender.training_history
            
            if (st.session_state.trained_lstm_recommender and 
                hasattr(st.session_state.trained_lstm_recommender, 'training_history')):
                lstm_history = st.session_state.trained_lstm_recommender.training_history
            
            if ncf_history or lstm_history:
                # 损失曲线
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📉 训练损失曲线**")
                    fig_loss = go.Figure()
                    
                    if ncf_history and ncf_history['epochs']:
                        fig_loss.add_trace(go.Scatter(
                            x=ncf_history['epochs'],
                            y=ncf_history['losses'],
                            mode='lines+markers',
                            name='NCF深度学习',
                            line=dict(color='#FF6B6B', width=3),
                            marker=dict(size=8)
                        ))
                    
                    if lstm_history and lstm_history['epochs']:
                        fig_loss.add_trace(go.Scatter(
                            x=lstm_history['epochs'],
                            y=lstm_history['losses'],
                            mode='lines+markers',
                            name='LSTM序列预测',
                            line=dict(color='#4ECDC4', width=3),
                            marker=dict(size=8)
                        ))
                    
                    fig_loss.update_layout(
                        title="训练损失变化",
                        xaxis_title="Epoch",
                        yaxis_title="损失值",
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                with col2:
                    st.write("**📈 训练准确率曲线**")
                    fig_acc = go.Figure()
                    
                    if ncf_history and ncf_history['epochs']:
                        fig_acc.add_trace(go.Scatter(
                            x=ncf_history['epochs'],
                            y=ncf_history['accuracies'],
                            mode='lines+markers',
                            name='NCF深度学习',
                            line=dict(color='#FF6B6B', width=3),
                            marker=dict(size=8)
                        ))
                    
                    if lstm_history and lstm_history['epochs']:
                        fig_acc.add_trace(go.Scatter(
                            x=lstm_history['epochs'],
                            y=lstm_history['accuracies'],
                            mode='lines+markers',
                            name='LSTM序列预测',
                            line=dict(color='#4ECDC4', width=3),
                            marker=dict(size=8)
                        ))
                    
                    fig_acc.update_layout(
                        title="训练准确率变化",
                        xaxis_title="Epoch",
                        yaxis_title="准确率 (%)",
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                # 训练进展分析
                st.subheader("📊 训练进展分析")
                col1, col2 = st.columns(2)
                
                with col1:
                    if ncf_history and len(ncf_history['losses']) > 1:
                        ncf_loss_improvement = ncf_history['losses'][0] - ncf_history['losses'][-1]
                        ncf_acc_improvement = ncf_history['accuracies'][-1] - ncf_history['accuracies'][0]
                        
                        st.metric(
                            "NCF损失改善",
                            f"{ncf_loss_improvement:.4f}",
                            delta=f"{(ncf_loss_improvement/ncf_history['losses'][0]*100):.1f}%"
                        )
                        st.metric(
                            "NCF准确率提升",
                            f"{ncf_acc_improvement:.2f}%",
                            delta=f"最终: {ncf_history['accuracies'][-1]:.2f}%"
                        )
                
                with col2:
                    if lstm_history and len(lstm_history['losses']) > 1:
                        lstm_loss_improvement = lstm_history['losses'][0] - lstm_history['losses'][-1]
                        lstm_acc_improvement = lstm_history['accuracies'][-1] - lstm_history['accuracies'][0]
                        
                        st.metric(
                            "LSTM损失改善",
                            f"{lstm_loss_improvement:.4f}",
                            delta=f"{(lstm_loss_improvement/lstm_history['losses'][0]*100):.1f}%"
                        )
                        st.metric(
                            "LSTM准确率提升",
                            f"{lstm_acc_improvement:.2f}%",
                            delta=f"最终: {lstm_history['accuracies'][-1]:.2f}%"
                        )
            else:
                st.warning("⚠️ 训练历史数据不可用")
        
        with tab2:
            st.subheader("模型性能雷达图比较")
            
            # 获取模型统计信息
            ncf_stats = None
            lstm_stats = None
            
            if (st.session_state.trained_ncf_recommender and 
                hasattr(st.session_state.trained_ncf_recommender, 'model_stats')):
                ncf_stats = st.session_state.trained_ncf_recommender.model_stats
            
            if (st.session_state.trained_lstm_recommender and 
                hasattr(st.session_state.trained_lstm_recommender, 'model_stats')):
                lstm_stats = st.session_state.trained_lstm_recommender.model_stats
            
            if ncf_stats and lstm_stats:
                # 收集实际数值用于显示
                raw_metrics = {}
                radar_values = {}
                
                # 1. 模型简洁性 (参数数量的倒数，越简洁越好)
                ncf_params = ncf_stats.get('total_params', 0)
                lstm_params = lstm_stats.get('total_params', 0)
                if ncf_params > 0 and lstm_params > 0:
                    # 计算简洁性分数 (越小的参数数量得分越高)
                    max_params = max(ncf_params, lstm_params)
                    ncf_simplicity = (max_params - ncf_params) / max_params * 100 + 10  # 最少给10分
                    lstm_simplicity = (max_params - lstm_params) / max_params * 100 + 10
                    
                    raw_metrics['模型简洁性'] = {
                        'NCF': f"{ncf_params:,} 参数",
                        'LSTM': f"{lstm_params:,} 参数"
                    }
                    radar_values['模型简洁性'] = {
                        'NCF': ncf_simplicity,
                        'LSTM': lstm_simplicity
                    }
                
                # 2. 训练数据规模 (样本数量)
                ncf_samples = ncf_stats.get('training_samples', 0)
                lstm_samples = lstm_stats.get('num_sequences', 0)
                if ncf_samples > 0 and lstm_samples > 0:
                    # 不直接拉满，用对数缩放
                    import math
                    ncf_data_score = min(90, math.log10(ncf_samples) * 15)  # 最高90分
                    lstm_data_score = min(90, math.log10(lstm_samples) * 15)
                    
                    raw_metrics['训练数据规模'] = {
                        'NCF': f"{ncf_samples:,} 样本",
                        'LSTM': f"{lstm_samples:,} 样本"
                    }
                    radar_values['训练数据规模'] = {
                        'NCF': ncf_data_score,
                        'LSTM': lstm_data_score
                    }
                
                # 3. 最终准确率
                if ncf_history and lstm_history:
                    if ncf_history.get('accuracies') and lstm_history.get('accuracies'):
                        ncf_final_acc = ncf_history['accuracies'][-1] if ncf_history['accuracies'] else 0
                        lstm_final_acc = lstm_history['accuracies'][-1] if lstm_history['accuracies'] else 0
                        
                        raw_metrics['训练准确率'] = {
                            'NCF': f"{ncf_final_acc:.1f}%",
                            'LSTM': f"{lstm_final_acc:.1f}%"
                        }
                        radar_values['训练准确率'] = {
                            'NCF': ncf_final_acc,  # 直接使用准确率百分比
                            'LSTM': lstm_final_acc
                        }
                
                # 4. 收敛速度 (损失下降程度百分比)
                if ncf_history and lstm_history:
                    if ncf_history.get('losses') and lstm_history.get('losses'):
                        ncf_convergence = 0
                        lstm_convergence = 0
                        
                        if len(ncf_history['losses']) > 1 and ncf_history['losses'][0] > 0:
                            ncf_convergence = (ncf_history['losses'][0] - ncf_history['losses'][-1]) / ncf_history['losses'][0] * 100
                        
                        if len(lstm_history['losses']) > 1 and lstm_history['losses'][0] > 0:
                            lstm_convergence = (lstm_history['losses'][0] - lstm_history['losses'][-1]) / lstm_history['losses'][0] * 100
                        
                        raw_metrics['收敛效果'] = {
                            'NCF': f"{ncf_convergence:.1f}% 损失下降",
                            'LSTM': f"{lstm_convergence:.1f}% 损失下降"
                        }
                        radar_values['收敛效果'] = {
                            'NCF': min(90, ncf_convergence),  # 最高90分
                            'LSTM': min(90, lstm_convergence)
                        }
                
                # 5. 用户覆盖率
                ncf_users = ncf_stats.get('num_users', 0)
                lstm_users = lstm_stats.get('valid_users', 0)
                total_users = len(self.data['user_id'].unique()) if self.data is not None else max(ncf_users, lstm_users)
                
                if ncf_users > 0 and lstm_users > 0 and total_users > 0:
                    ncf_coverage = (ncf_users / total_users) * 100
                    lstm_coverage = (lstm_users / total_users) * 100
                    
                    raw_metrics['用户覆盖率'] = {
                        'NCF': f"{ncf_coverage:.1f}% ({ncf_users:,}用户)",
                        'LSTM': f"{lstm_coverage:.1f}% ({lstm_users:,}用户)"
                    }
                    radar_values['用户覆盖率'] = {
                        'NCF': ncf_coverage,
                        'LSTM': lstm_coverage
                    }
                
                # 6. 数据稀疏性适应度 (稀疏度越高，适应性要求越高)
                ncf_sparsity = ncf_stats.get('sparsity', 0)
                if ncf_sparsity > 0:
                    # 稀疏度适应性：稀疏数据下的表现能力
                    sparsity_challenge = ncf_sparsity * 1000000  # 将稀疏度放大
                    ncf_sparsity_score = min(85, 20 + (1 - ncf_sparsity * 1000) * 65) if ncf_sparsity < 0.001 else 20
                    lstm_sparsity_score = min(75, 15 + (1 - ncf_sparsity * 1000) * 60) if ncf_sparsity < 0.001 else 15  # LSTM在极稀疏数据下表现相对较差
                    
                    raw_metrics['稀疏数据适应'] = {
                        'NCF': f"稀疏度 {ncf_sparsity:.6f}",
                        'LSTM': f"稀疏度 {ncf_sparsity:.6f}"
                    }
                    radar_values['稀疏数据适应'] = {
                        'NCF': ncf_sparsity_score,
                        'LSTM': lstm_sparsity_score
                    }
                
                # 创建雷达图
                if radar_values:
                    categories = list(radar_values.keys())
                    ncf_values = [radar_values[cat]['NCF'] for cat in categories]
                    lstm_values = [radar_values[cat]['LSTM'] for cat in categories]
                    
                    # 显示具体数值表格
                    st.subheader("📊 模型性能具体指标")
                    
                    # 创建对比表格
                    comparison_data = []
                    for metric in categories:
                        if metric in raw_metrics:
                            comparison_data.append({
                                '性能指标': metric,
                                'NCF深度学习': raw_metrics[metric]['NCF'],
                                'LSTM序列预测': raw_metrics[metric]['LSTM'],
                                'NCF得分': f"{radar_values[metric]['NCF']:.1f}",
                                'LSTM得分': f"{radar_values[metric]['LSTM']:.1f}"
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    # 绘制雷达图
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=ncf_values + [ncf_values[0]],  # 闭合图形
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='NCF深度学习',
                        line_color='#FF6B6B',
                        fillcolor='rgba(255, 107, 107, 0.3)',
                        hovertemplate='<b>NCF深度学习</b><br>' +
                                    '%{theta}: %{r:.1f}分<br>' +
                                    '<extra></extra>'
                    ))
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=lstm_values + [lstm_values[0]],  # 闭合图形
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='LSTM序列预测',
                        line_color='#4ECDC4',
                        fillcolor='rgba(78, 205, 196, 0.3)',
                        hovertemplate='<b>LSTM序列预测</b><br>' +
                                    '%{theta}: %{r:.1f}分<br>' +
                                    '<extra></extra>'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                tickvals=[20, 40, 60, 80, 100],
                                ticktext=['20', '40', '60', '80', '100'],
                                gridcolor='lightgray'
                            ),
                            angularaxis=dict(
                                gridcolor='lightgray'
                            )
                        ),
                        showlegend=True,
                        title="模型性能雷达图比较 (评分范围: 0-100)",
                        height=600,
                        font=dict(size=12)
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # 雷达图说明
                    st.info("""
                    **雷达图评分说明:**
                    - **模型简洁性**: 参数越少得分越高 (轻量化程度)
                    - **训练数据规模**: 训练样本数量 (对数缩放，最高90分)
                    - **训练准确率**: 最终训练准确率百分比 (直接使用%)
                    - **收敛效果**: 训练过程中损失下降百分比 (最高90分)
                    - **用户覆盖率**: 可推荐用户占总用户的百分比
                    - **稀疏数据适应**: 在稀疏数据场景下的表现能力
                    
                    **注**: 得分越高表示该指标表现越好，满分100分
                    """)
                else:
                    st.warning("⚠️ 无法生成雷达图：缺少必要的统计数据")
            else:
                st.warning("⚠️ 雷达图不可用：缺少模型统计信息")
        
        with tab3:
            st.subheader("详细统计信息")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🧠 NCF深度学习模型**")
                if st.session_state.trained_ncf_recommender:
                    ncf_model = st.session_state.trained_ncf_recommender
                    
                    # 基本信息
                    if hasattr(ncf_model, 'model_stats'):
                        stats = ncf_model.model_stats
                        st.info(f"""
                        **基本信息:**
                        - 用户数: {stats.get('num_users', 'N/A'):,}
                        - 商品数: {stats.get('num_items', 'N/A'):,}
                        - 训练样本数: {stats.get('training_samples', 'N/A'):,}
                        - 数据稀疏度: {stats.get('sparsity', 'N/A'):.6f}
                        
                        **模型参数:**
                        - 总参数数: {stats.get('total_params', 'N/A'):,}
                        - 可训练参数: {stats.get('trainable_params', 'N/A'):,}
                        """)
                    
                    # 训练结果
                    if hasattr(ncf_model, 'training_history') and ncf_model.training_history['epochs']:
                        history = ncf_model.training_history
                        st.success(f"""
                        **训练结果:**
                        - 训练轮数: {len(history['epochs'])}
                        - 初始损失: {history['losses'][0]:.6f}
                        - 最终损失: {history['losses'][-1]:.6f}
                        - 最终准确率: {history['accuracies'][-1]:.2f}%
                        """)
                else:
                    st.warning("NCF模型未训练")
            
            with col2:
                st.write("**📈 LSTM序列预测模型**")
                if st.session_state.trained_lstm_recommender:
                    lstm_model = st.session_state.trained_lstm_recommender
                    
                    # 基本信息
                    if hasattr(lstm_model, 'model_stats'):
                        stats = lstm_model.model_stats
                        st.info(f"""
                        **基本信息:**
                        - 有效用户数: {stats.get('valid_users', 'N/A'):,}
                        - 商品词汇量: {stats.get('vocab_size_item', 'N/A'):,}
                        - 类别词汇量: {stats.get('vocab_size_cat', 'N/A'):,}
                        - 训练序列数: {stats.get('num_sequences', 'N/A'):,}
                        
                        **模型参数:**
                        - 总参数数: {stats.get('total_params', 'N/A'):,}
                        - 可训练参数: {stats.get('trainable_params', 'N/A'):,}
                        """)
                    
                    # 训练结果
                    if hasattr(lstm_model, 'training_history') and lstm_model.training_history['epochs']:
                        history = lstm_model.training_history
                        st.success(f"""
                        **训练结果:**
                        - 训练轮数: {len(history['epochs'])}
                        - 初始损失: {history['losses'][0]:.6f}
                        - 最终损失: {history['losses'][-1]:.6f}
                        - 最终准确率: {history['accuracies'][-1]:.2f}%
                        """)
                else:
                    st.warning("LSTM模型未训练")
    
    def render_personalized_recommendation(self):
        """渲染个性化推荐页面"""
        st.title("🎯 个性化推荐")
        
        if self.data is None:
            st.warning("⚠️ 请先在侧边栏上传数据文件")
            return
        
        # 检查模型训练状态
        if not st.session_state.models_trained:
            st.warning("⚠️ 请先在 '推荐算法比较' 页面训练模型")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **训练步骤:**
                1. 点击侧边栏选择 '推荐算法比较'
                2. 点击 '开始训练模型' 按钮
                3. 等待训练完成
                4. 返回此页面进行推荐
                """)
            
            with col2:
                st.image("data:image/svg+xml,%3csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3e%3ctext x='50' y='50' font-size='50' text-anchor='middle' dy='.3em'%3e🤖%3c/text%3e%3c/svg%3e", width=100)
                st.write("**模型训练中...**")
            
            return
        
        # 用户选择
        st.subheader("👤 选择用户")
        

                
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 智能用户选择：只显示可以推荐的用户
            if st.session_state.models_trained:
                # 获取所有模型中可推荐的用户
                ncf_users = set()
                
                if (st.session_state.trained_ncf_recommender and 
                    hasattr(st.session_state.trained_ncf_recommender, 'user2idx')):
                    ncf_users = set(st.session_state.trained_ncf_recommender.user2idx.keys())
                
                # 使用NCF用户作为可推荐用户
                available_users = ncf_users
                
                if available_users:
                    # 限制用户列表长度，并排序
                    user_list = sorted(list(available_users))[:200]
                    st.success(f"✅ 找到 {len(available_users):,} 个可推荐用户 (显示前200个)")
                    
                    # 用户选择方式
                    selection_method = st.radio(
                        "选择用户方式",
                        ["从列表选择", "输入用户ID"]
                    )
                    
                    if selection_method == "从列表选择":
                        selected_user = st.selectbox("选择用户ID", user_list)
                    else:
                        input_user = st.text_input("输入用户ID", placeholder="请输入一个用户ID")
                        if input_user:
                            try:
                                # 尝试转换为数字（如果是数字字符串）
                                input_user_parsed = int(input_user) if input_user.isdigit() else input_user
                                if input_user_parsed in available_users:
                                    selected_user = input_user_parsed
                                    st.success(f"✅ 用户 {input_user_parsed} 可以推荐")
                                else:
                                    st.error(f"❌ 用户 {input_user_parsed} 不在训练数据中")
                                    st.info("可推荐的用户示例: " + ", ".join(map(str, user_list[:5])))
                                    selected_user = user_list[0] if user_list else None
                            except ValueError:
                                st.error("请输入有效的用户ID")
                                selected_user = user_list[0] if user_list else None
                        else:
                            selected_user = user_list[0] if user_list else None
                            
                    # 显示用户在各模型中的状态
                    if selected_user:
                        status_info = f"**用户 {selected_user} 状态:**\n"
                        status_info += f"- NCF模型: {'✅ 可推荐' if selected_user in ncf_users else '❌ 不可用'}"
                        st.info(status_info)
                else:
                    st.error("❌ 没有找到可推荐的用户")
                    st.error("**可能的原因:**")
                    st.error("- 数据中没有购买行为记录")
                    st.error("- 模型训练失败或数据不足")
                    st.error("- 训练数据过滤太严格")
                    
                    # 提供调试信息
                    st.info("**调试信息:**")
                    st.info(f"- NCF模型用户数: {len(ncf_users)}")
                    
                    selected_user = None
            else:
                # 如果模型未训练，显示有购买行为的用户作为预览
                users_with_purchases = self.data[self.data['behavior_type'] == 'buy']['user_id'].unique()
                
                if len(users_with_purchases) == 0:
                    st.error("❌ 数据中没有购买行为记录")
                    st.info("推荐系统需要购买行为数据进行训练")
                    selected_user = None
                else:
                    user_list = sorted(users_with_purchases[:100])
                    selected_user = st.selectbox("选择用户ID (模型未训练)", user_list)
                    st.warning("⚠️ 模型未训练，请先在'推荐算法比较'页面训练模型")
            
            if selected_user is not None:
                # 推荐算法选择
                algorithm = st.selectbox(
                    "选择推荐算法",
                    ["NCF深度学习", "LSTM序列预测"]
                )
                
                recommendation_count = st.slider("推荐数量", 5, 20, 10)
                
        
        with col2:
            # 用户历史行为分析
            st.write("**用户历史行为分析**")
            user_history = self.data[self.data['user_id'] == selected_user].tail(20)
            
            if len(user_history) > 0:
                # 行为统计
                behavior_summary = user_history['behavior_type'].value_counts()
                
                # 创建行为分析的子图
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('行为类型分布', '购买品类分布'),
                    specs=[[{"type": "pie"}, {"type": "pie"}]]
                )
                
                # 行为类型饼图
                fig.add_trace(
                    go.Pie(
                        labels=behavior_summary.index,
                        values=behavior_summary.values,
                        name="行为类型"
                    ),
                    row=1, col=1
                )
                
                # 购买品类分布
                if 'category_id' in user_history.columns:
                    category_summary = user_history[user_history['behavior_type'] == 'buy']['category_id'].value_counts().head(5)
                    if len(category_summary) > 0:
                        fig.add_trace(
                            go.Pie(
                                labels=category_summary.index,
                                values=category_summary.values,
                                name="购买品类"
                            ),
                            row=1, col=2
                        )
                
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # 用户行为时间序列
                if 'timestamp_dt' in user_history.columns or 'date' in user_history.columns:
                    time_col = 'timestamp_dt' if 'timestamp_dt' in user_history.columns else 'date'
                    
                    try:
                        # 按日期统计行为数量
                        if time_col == 'timestamp_dt':
                            # 确保timestamp_dt是datetime类型
                            if not pd.api.types.is_datetime64_any_dtype(user_history[time_col]):
                                user_history[time_col] = pd.to_datetime(user_history[time_col], errors='coerce')
                            
                            # 如果转换成功，创建date列
                            if pd.api.types.is_datetime64_any_dtype(user_history[time_col]):
                                user_history['date'] = user_history[time_col].dt.date
                                time_series = user_history.groupby('date').size()
                            else:
                                # 如果转换失败，使用原始列
                                time_series = user_history.groupby(time_col).size()
                        else:
                            # 对于date列，尝试确保是正确的格式
                            if not pd.api.types.is_datetime64_any_dtype(user_history[time_col]):
                                try:
                                    user_history[time_col] = pd.to_datetime(user_history[time_col], errors='coerce')
                                except:
                                    pass  # 如果转换失败，继续使用原始数据
                            time_series = user_history.groupby(time_col).size()
                        
                        # 只有当time_series不为空时才绘制图表
                        if len(time_series) > 0:
                            fig = px.line(
                                x=time_series.index,
                                y=time_series.values,
                                title="用户行为时间序列",
                                labels={'x': '日期', 'y': '行为次数'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("该用户的时间序列数据不足以绘制图表")
                            
                    except Exception as e:
                        st.warning(f"时间序列图表生成失败: {str(e)}")
                        st.info("跳过时间序列分析，继续显示其他信息")
            
            # 用户特征摘要
            st.subheader("📊 用户特征摘要")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                total_actions = len(user_history)
                st.metric("总行为数", total_actions)
            
            with col_b:
                unique_items = user_history['item_id'].nunique()
                st.metric("浏览商品数", unique_items)
            
            with col_c:
                purchase_count = len(user_history[user_history['behavior_type'] == 'buy'])
                st.metric("购买次数", purchase_count)
        
        # 生成推荐结果
        st.subheader("📋 推荐结果")
        
        if st.button("🎯 生成个性化推荐", type="primary"):
            with st.spinner(f"正在使用{algorithm}算法生成个性化推荐..."):
                
                try:
                    recommendations = []
                    debug_info = []
                    
                    st.info(f"开始为用户 {selected_user} 生成推荐...")
                    
                    if algorithm == "NCF深度学习":
                        if st.session_state.trained_ncf_recommender is not None:
                            st.info("使用训练好的NCF深度学习模型...")
                            ncf_results = st.session_state.trained_ncf_recommender.recommend(selected_user, recommendation_count)
                            debug_info.append(f"NCF模型返回结果类型: {type(ncf_results)}")
                            debug_info.append(f"NCF模型返回结果长度: {len(ncf_results)}")
                            
                            if len(ncf_results) > 0:
                                recommendations = ncf_results
                                debug_info.append(f"NCF推荐结果: {recommendations[:3]}...")  # 显示前3个
                            else:
                                debug_info.append("NCF模型返回空结果")
                        else:
                            debug_info.append("NCF模型未训练或不可用")
                    
                    elif algorithm == "LSTM序列预测":
                        if st.session_state.trained_lstm_recommender is not None:
                            st.info("使用训练好的LSTM序列预测模型...")
                            lstm_results = st.session_state.trained_lstm_recommender.recommend(selected_user, self.data, recommendation_count)
                            debug_info.append(f"LSTM模型返回结果类型: {type(lstm_results)}")
                            debug_info.append(f"LSTM模型返回结果长度: {len(lstm_results)}")
                            
                            if len(lstm_results) > 0:
                                recommendations = lstm_results
                                debug_info.append(f"LSTM推荐结果: {lstm_results[:3]}...")  # 显示前3个
                            else:
                                debug_info.append("LSTM模型返回空结果")
                        else:
                            debug_info.append("LSTM模型未训练或不可用")
                    
                    elif algorithm == "混合推荐":
                        st.info("使用混合推荐模型...")
                        # 结合两种算法的结果
                        cf_results = []
                        ncf_results = []
                        lstm_results = []
                        
                        if st.session_state.trained_cf_recommender is not None:
                            cf_res = st.session_state.trained_cf_recommender.recommend(selected_user, recommendation_count)
                            debug_info.append(f"混合推荐 - CF结果长度: {len(cf_res)}")
                            if len(cf_res) > 0:
                                cf_results = [(item_id, score * 0.4) for item_id, score in zip(cf_res.index, cf_res.values)]
                        
                        if st.session_state.trained_ncf_recommender is not None:
                            ncf_res = st.session_state.trained_ncf_recommender.recommend(selected_user, recommendation_count)
                            debug_info.append(f"混合推荐 - NCF结果长度: {len(ncf_res)}")
                            if len(ncf_res) > 0:
                                ncf_results = [(item_id, score * 0.6) for item_id, score in ncf_res]
                        
                        if st.session_state.trained_lstm_recommender is not None:
                            lstm_res = st.session_state.trained_lstm_recommender.recommend(selected_user, self.data, recommendation_count)
                            debug_info.append(f"混合推荐 - LSTM结果长度: {len(lstm_res)}")
                            if len(lstm_res) > 0:
                                lstm_results = [(item_id, score * 0.2) for item_id, score in lstm_res]
                        
                        # 合并并排序
                        all_results = {}
                        for item_id, score in cf_results + ncf_results + lstm_results:
                            if item_id in all_results:
                                all_results[item_id] += score
                            else:
                                all_results[item_id] = score
                        
                        recommendations = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:recommendation_count]
                        debug_info.append(f"混合推荐最终结果数量: {len(recommendations)}")
                    
                    # 显示调试信息
                    with st.expander("🔍 调试信息"):
                        for info in debug_info:
                            st.write(f"- {info}")
                    
                    if recommendations and len(recommendations) > 0:
                        # 构建推荐结果数据框
                        recommendations_df = pd.DataFrame(recommendations, columns=['商品ID', '推荐分数'])
                        
                        # 添加商品额外信息
                        item_info = []
                        for item_id in recommendations_df['商品ID']:
                            item_data = self.data[self.data['item_id'] == item_id]
                            if len(item_data) > 0:
                                category = item_data['category_id'].iloc[0] if 'category_id' in item_data.columns else 'Unknown'
                                popularity = len(item_data)
                                item_info.append({
                                    '商品ID': item_id,
                                    '类别': category,
                                    '热度': popularity
                                })
                        
                        item_info_df = pd.DataFrame(item_info)
                        if len(item_info_df) > 0:
                            recommendations_df = recommendations_df.merge(item_info_df, on='商品ID', how='left')
                        
                        # 显示推荐结果
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("🏆 推荐商品列表")
                            
                            # 添加排名列
                            recommendations_df['排名'] = range(1, len(recommendations_df) + 1)
                            display_df = recommendations_df[['排名', '商品ID', '推荐分数', '类别', '热度']]
                            
                            # 格式化推荐分数
                            display_df['推荐分数'] = display_df['推荐分数'].apply(lambda x: f"{x:.4f}")
                            
                            st.dataframe(display_df, use_container_width=True)
                        
                        with col2:
                            # 推荐分数分布
                            fig = px.bar(
                                recommendations_df.head(10),
                                x='商品ID',
                                y='推荐分数',
                                title="Top 10 推荐分数",
                                color='推荐分数',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(xaxis_tickangle=45, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 推荐多样性分析
                        if 'category_id' in self.data.columns and '类别' in recommendations_df.columns:
                            st.subheader("🎨 推荐多样性分析")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 推荐品类分布
                                category_dist = recommendations_df['类别'].value_counts()
                                fig = px.pie(
                                    values=category_dist.values,
                                    names=category_dist.index,
                                    title="推荐商品类别分布"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # 热度vs分数散点图
                                if '热度' in recommendations_df.columns:
                                    fig = px.scatter(
                                        recommendations_df,
                                        x='热度',
                                        y='推荐分数',
                                        color='类别',
                                        title="商品热度 vs 推荐分数",
                                        hover_data=['商品ID']
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # 推荐解释
                        st.subheader("💡 推荐解释")
                        
                        explanation_text = f"""
                        **推荐算法:** {algorithm}
                        
                        **推荐依据:**
                        """
                        
                        if algorithm == "协同过滤":
                            explanation_text += """
                            - 基于与您相似的用户的购买行为
                            - 分析用户之间的相似度模式
                            - 推荐相似用户喜欢的商品
                            """
                        elif algorithm == "NCF深度学习":
                            explanation_text += """
                            - 使用深度神经网络学习用户-商品复杂关系
                            - 考虑用户和商品的高维特征表示
                            - 通过非线性变换捕获潜在偏好
                            """
                        elif algorithm == "LSTM序列预测":
                            explanation_text += """
                            - 考虑时间序列特征
                            - 能捕获用户行为模式
                            - 适合序列推荐
                            """
                        elif algorithm == "混合推荐":
                            explanation_text += """
                            - 结合协同过滤和深度学习的优势
                            - 协同过滤权重: 40%, 深度学习权重: 60%
                            - 提供更加稳定和准确的推荐结果
                            """
                        
                        explanation_text += f"""
                        
                        **个性化特征:**
                        - 用户历史行为: {len(user_history)} 条记录
                        - 购买行为: {purchase_count} 次
                        - 浏览商品: {unique_items} 种
                        - 推荐商品均为用户未曾交互过的商品
                        """
                        
                        st.info(explanation_text)
                        
                        # 推荐效果预测
                        st.subheader("📈 推荐效果预测")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_score = recommendations_df['推荐分数'].mean()
                            st.metric("平均推荐分数", f"{avg_score:.4f}")
                        
                        with col2:
                            if '类别' in recommendations_df.columns:
                                diversity_score = len(recommendations_df['类别'].unique()) / len(recommendations_df)
                                st.metric("多样性指数", f"{diversity_score:.2f}")
                        
                        with col3:
                            if '热度' in recommendations_df.columns:
                                avg_popularity = recommendations_df['热度'].mean()
                                st.metric("平均商品热度", f"{avg_popularity:.0f}")
                    
                    else:
                        st.error("无法为该用户生成推荐，可能的原因：")
                        st.write("- 用户没有足够的历史数据")
                        st.write("- 模型训练数据中没有该用户")
                        st.write("- 所有候选商品都已被用户交互过")
                
                except Exception as e:
                    st.error(f"推荐生成失败: {str(e)}")
                    st.info("建议检查数据格式或重新训练模型")
    
    def train_models(self):
        """训练推荐模型"""
        if self.data is None or st.session_state.models_trained:
            return
            
        st.info("🔄 正在训练推荐模型，请稍候...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 使用数据子集以提高训练速度
            sample_size = min(50000, len(self.data))
            df_sample = self.data.sample(n=sample_size, random_state=42)
            
            # 训练NCF模型
            status_text.text("训练NCF深度学习模型...")
            progress_bar.progress(50)
            self.ncf_recommender.fit(df_sample, epochs=3)  # 减少epoch数以提高速度
            # 保存到session_state
            st.session_state.trained_ncf_recommender = self.ncf_recommender
            
            # 训练LSTM模型
            status_text.text("训练LSTM序列预测模型...")
            progress_bar.progress(100)
            self.lstm_recommender.fit(df_sample)
            # 保存到session_state
            st.session_state.trained_lstm_recommender = self.lstm_recommender
            
            progress_bar.progress(100)
            status_text.text("模型训练完成！")
            # 更新训练状态
            st.session_state.models_trained = True
            
            st.success("✅ 所有推荐模型训练完成！")
            
        except Exception as e:
            st.error(f"模型训练失败: {str(e)}")
            st.info("建议使用较小的数据样本进行训练")
    
    def run(self):
        """运行仪表板"""
        # 渲染侧边栏
        analysis_type = self.render_sidebar()
        
        # 根据选择渲染不同页面
        if analysis_type == "数据概览":
            self.render_data_overview()
        elif analysis_type == "用户行为分析":
            self.render_user_behavior_analysis()
        elif analysis_type == "用户画像分析":
            self.render_user_segmentation()
        elif analysis_type == "推荐算法比较":
            self.render_algorithm_comparison()
        elif analysis_type == "个性化推荐":
            self.render_personalized_recommendation()

def main():
    """主函数"""
    dashboard = RecommendationDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 