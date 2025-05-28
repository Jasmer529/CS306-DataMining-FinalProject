"""
基于图神经网络的推荐系统
使用用户-商品二部图和GCN进行推荐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class GraphConvolution(nn.Module):
    """图卷积层"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        前向传播
        :param x: 节点特征 [num_nodes, input_dim]
        :param adj: 邻接矩阵 [num_nodes, num_nodes]
        """
        # 特征变换
        h = self.linear(x)
        # 图卷积：聚合邻居信息
        output = torch.spmm(adj, h)
        output = self.dropout(output)
        return F.relu(output)

class GraphNeuralRecommender(nn.Module):
    """图神经网络推荐系统"""
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=32, num_layers=2, dropout=0.1):
        super(GraphNeuralRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 用户和商品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 图卷积层
        self.gc_layers = nn.ModuleList()
        input_dim = embedding_dim
        
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else embedding_dim
            self.gc_layers.append(GraphConvolution(input_dim, output_dim, dropout))
            input_dim = output_dim
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, user_ids, item_ids, adj_matrix):
        """
        前向传播
        :param user_ids: 用户ID列表
        :param item_ids: 商品ID列表  
        :param adj_matrix: 图邻接矩阵
        """
        # 获取初始嵌入
        user_emb = self.user_embedding.weight  # [num_users, embedding_dim]
        item_emb = self.item_embedding.weight  # [num_items, embedding_dim]
        
        # 拼接用户和商品嵌入形成图节点特征
        node_features = torch.cat([user_emb, item_emb], dim=0)  # [num_users + num_items, embedding_dim]
        
        # 图卷积传播
        h = node_features
        for gc_layer in self.gc_layers:
            h = gc_layer(h, adj_matrix)
        
        # 分离用户和商品的最终嵌入
        final_user_emb = h[:self.num_users]  # [num_users, embedding_dim]
        final_item_emb = h[self.num_users:]  # [num_items, embedding_dim]
        
        # 获取特定用户和商品的嵌入
        batch_user_emb = final_user_emb[user_ids]  # [batch_size, embedding_dim]
        batch_item_emb = final_item_emb[item_ids]  # [batch_size, embedding_dim]
        
        # 特征拼接
        combined_features = torch.cat([batch_user_emb, batch_item_emb], dim=1)
        
        # 预测评分
        scores = self.predictor(combined_features).squeeze()
        
        return scores, final_user_emb, final_item_emb

class GNNRecommenderSystem:
    """GNN推荐系统完整实现"""
    
    def __init__(self, embedding_dim=64, hidden_dim=32, num_layers=2, learning_rate=0.001):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.adj_matrix = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 初始化GNN推荐系统 (设备: {self.device})")
    
    def prepare_data(self, data):
        """准备训练数据"""
        print("📊 准备GNN训练数据...")
        
        # 创建用户和商品的映射
        unique_users = data['user_id'].unique()
        unique_items = data['item_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        print(f"   用户数量: {self.num_users}")
        print(f"   商品数量: {self.num_items}")
        
        # 创建邻接矩阵
        self._create_adjacency_matrix(data)
        
        # 准备训练样本
        train_data = self._prepare_training_samples(data)
        
        return train_data
    
    def _create_adjacency_matrix(self, data):
        """创建用户-商品二部图的邻接矩阵"""
        print("🕸️ 构建用户-商品二部图...")
        
        total_nodes = self.num_users + self.num_items
        
        # 创建边列表
        rows, cols = [], []
        
        for _, row in data.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']] + self.num_users  # 商品索引偏移
            
            # 添加用户-商品边（无向图）
            rows.extend([user_idx, item_idx])
            cols.extend([item_idx, user_idx])
        
        # 创建稀疏邻接矩阵
        adj_coo = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), 
            shape=(total_nodes, total_nodes)
        )
        
        # 归一化邻接矩阵
        adj_coo = self._normalize_adjacency(adj_coo)
        
        # 转换为PyTorch张量
        indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
        values = torch.FloatTensor(adj_coo.data)
        shape = adj_coo.shape
        
        self.adj_matrix = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
        
        print(f"   图节点数: {total_nodes}")
        print(f"   图边数: {len(adj_coo.data)}")
    
    def _normalize_adjacency(self, adj):
        """归一化邻接矩阵"""
        # 计算度矩阵
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = coo_matrix((d_inv_sqrt, (range(len(d_inv_sqrt)), range(len(d_inv_sqrt)))), shape=adj.shape)
        
        # 对称归一化: D^(-1/2) * A * D^(-1/2)
        normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        return normalized_adj.tocoo()
    
    def _prepare_training_samples(self, data):
        """准备训练样本"""
        # 正样本：用户实际交互的商品
        positive_samples = []
        for _, row in data.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            
            # 根据行为类型设置标签权重
            if row['behavior_type'] == 'buy':
                label = 1.0
            elif row['behavior_type'] == 'cart':
                label = 0.8
            elif row['behavior_type'] == 'fav':
                label = 0.6
            else:  # pv
                label = 0.4
            
            positive_samples.append((user_idx, item_idx, label))
        
        # 负样本：随机选择用户未交互的商品
        negative_samples = []
        user_items = defaultdict(set)
        
        # 记录每个用户交互过的商品
        for _, row in data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            user_items[user_id].add(item_id)
        
        # 为每个正样本生成一个负样本
        all_items = set(self.item_to_idx.keys())
        for user_idx, item_idx, _ in positive_samples:
            user_id = self.idx_to_user[user_idx]
            uninteracted_items = all_items - user_items[user_id]
            
            if uninteracted_items:
                neg_item_id = np.random.choice(list(uninteracted_items))
                neg_item_idx = self.item_to_idx[neg_item_id]
                negative_samples.append((user_idx, neg_item_idx, 0.0))
        
        # 合并正负样本
        all_samples = positive_samples + negative_samples
        np.random.shuffle(all_samples)
        
        print(f"   训练样本数: {len(all_samples)} (正样本: {len(positive_samples)}, 负样本: {len(negative_samples)})")
        
        return all_samples
    
    def train(self, data, epochs=50, batch_size=1024, validation_split=0.1):
        """训练GNN模型"""
        print(f"\n🚀 开始训练GNN模型 (epochs={epochs}, batch_size={batch_size})")
        
        # 准备数据
        train_samples = self.prepare_data(data)
        
        # 初始化模型
        self.model = GraphNeuralRecommender(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # 数据分割
        split_idx = int(len(train_samples) * (1 - validation_split))
        train_data = train_samples[:split_idx]
        val_data = train_samples[split_idx:]
        
        losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            num_batches = len(train_data) // batch_size + 1
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                user_ids = torch.LongTensor([sample[0] for sample in batch]).to(self.device)
                item_ids = torch.LongTensor([sample[1] for sample in batch]).to(self.device)
                labels = torch.FloatTensor([sample[2] for sample in batch]).to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                scores, _, _ = self.model(user_ids, item_ids, self.adj_matrix)
                
                # 计算损失
                loss = criterion(scores, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / num_batches
            losses.append(avg_train_loss)
            
            # 验证阶段
            if val_data:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_data), batch_size):
                        batch = val_data[i:i+batch_size]
                        
                        user_ids = torch.LongTensor([sample[0] for sample in batch]).to(self.device)
                        item_ids = torch.LongTensor([sample[1] for sample in batch]).to(self.device)
                        labels = torch.FloatTensor([sample[2] for sample in batch]).to(self.device)
                        
                        scores, _, _ = self.model(user_ids, item_ids, self.adj_matrix)
                        loss = criterion(scores, labels)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        print("✅ GNN模型训练完成!")
        return losses, val_losses
    
    def recommend_for_user(self, user_id, top_k=10, exclude_seen=True):
        """为指定用户生成推荐"""
        if self.model is None:
            print("❌ 请先训练模型")
            return []
        
        if user_id not in self.user_to_idx:
            print(f"❌ 用户 {user_id} 不在训练数据中")
            return []
        
        self.model.eval()
        
        with torch.no_grad():
            user_idx = self.user_to_idx[user_id]
            
            # 为该用户对所有商品进行评分
            user_ids = torch.LongTensor([user_idx] * self.num_items).to(self.device)
            item_ids = torch.LongTensor(list(range(self.num_items))).to(self.device)
            
            scores, _, _ = self.model(user_ids, item_ids, self.adj_matrix)
            scores = scores.cpu().numpy()
            
            # 排序并选择Top-K
            item_scores = [(self.idx_to_item[i], scores[i]) for i in range(self.num_items)]
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 可选：排除用户已经交互过的商品
            if exclude_seen:
                # 这里需要额外的逻辑来排除已交互商品
                pass
            
            recommendations = [item_id for item_id, score in item_scores[:top_k]]
            
        return recommendations
    
    def get_embeddings(self):
        """获取学习到的用户和商品嵌入"""
        if self.model is None:
            print("❌ 请先训练模型")
            return None, None
        
        self.model.eval()
        
        with torch.no_grad():
            # 使用所有用户和商品的索引
            all_user_ids = torch.LongTensor(list(range(self.num_users))).to(self.device)
            all_item_ids = torch.LongTensor(list(range(self.num_items))).to(self.device)
            
            # 通过模型获取最终嵌入
            _, user_embeddings, item_embeddings = self.model(
                all_user_ids, all_item_ids, self.adj_matrix
            )
            
            user_embeddings = user_embeddings.cpu().numpy()
            item_embeddings = item_embeddings.cpu().numpy()
        
        return user_embeddings, item_embeddings
    
    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            print("❌ 没有训练好的模型可以保存")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers
            }
        }, path)
        
        print(f"💾 模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 恢复配置
        self.user_to_idx = checkpoint['user_to_idx']
        self.item_to_idx = checkpoint['item_to_idx']
        self.idx_to_user = checkpoint['idx_to_user']
        self.idx_to_item = checkpoint['idx_to_item']
        self.num_users = checkpoint['num_users']
        self.num_items = checkpoint['num_items']
        
        config = checkpoint['config']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        
        # 重建模型
        self.model = GraphNeuralRecommender(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"�� 模型已从 {path} 加载") 