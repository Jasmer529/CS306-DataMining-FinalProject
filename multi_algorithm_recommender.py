"""
多算法推荐系统模块
包含协同过滤、矩阵分解、深度学习等多种推荐算法的实现和比较
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CollaborativeFiltering:
    """协同过滤推荐算法"""
    
    def __init__(self, method='user_based'):
        """
        初始化协同过滤
        :param method: 'user_based' 或 'item_based'
        """
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None
        
    def prepare_data(self, data):
        """准备用户-物品评分矩阵"""
        print(f"🔍 准备{self.method}协同过滤数据...")
        
        # 创建用户-物品交互矩阵 (用行为次数作为评分)
        interaction_counts = data.groupby(['user_id', 'item_id']).size().reset_index(name='rating')
        
        # 创建评分矩阵
        self.user_item_matrix = interaction_counts.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        print(f"✅ 评分矩阵构建完成: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def calculate_similarity(self):
        """计算相似度矩阵"""
        print(f"🔍 计算{self.method}相似度矩阵...")
        
        if self.method == 'user_based':
            # 用户相似度
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            self.similarity_df = pd.DataFrame(
                self.similarity_matrix,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
        else:
            # 物品相似度
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            self.similarity_df = pd.DataFrame(
                self.similarity_matrix,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        
        print(f"✅ 相似度矩阵计算完成: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def recommend(self, user_id, top_k=10):
        """为指定用户推荐物品"""
        if self.similarity_matrix is None:
            self.calculate_similarity()
            
        if user_id not in self.user_item_matrix.index:
            return []
        
        if self.method == 'user_based':
            return self._user_based_recommend(user_id, top_k)
        else:
            return self._item_based_recommend(user_id, top_k)
    
    def _user_based_recommend(self, user_id, top_k):
        """基于用户的协同过滤推荐"""
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.similarity_df.iloc[user_idx].drop(user_id)
        
        # 获取相似用户
        similar_users = user_similarities.nlargest(50).index
        
        # 计算推荐分数
        recommendations = {}
        user_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        for item in self.user_item_matrix.columns:
            if item not in user_items:  # 只推荐用户未交互过的物品
                score = 0
                sim_sum = 0
                
                for similar_user in similar_users:
                    if self.user_item_matrix.loc[similar_user, item] > 0:
                        similarity = user_similarities[similar_user]
                        rating = self.user_item_matrix.loc[similar_user, item]
                        score += similarity * rating
                        sim_sum += abs(similarity)
                
                if sim_sum > 0:
                    recommendations[item] = score / sim_sum
        
        # 返回top-k推荐
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recs[:top_k]]
    
    def _item_based_recommend(self, user_id, top_k):
        """基于物品的协同过滤推荐"""
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = user_items[user_items > 0].index
        
        recommendations = {}
        
        for item in self.user_item_matrix.columns:
            if item not in user_interacted_items:
                score = 0
                sim_sum = 0
                
                for interacted_item in user_interacted_items:
                    if item in self.similarity_df.index and interacted_item in self.similarity_df.columns:
                        similarity = self.similarity_df.loc[item, interacted_item]
                        rating = user_items[interacted_item]
                        score += similarity * rating
                        sim_sum += abs(similarity)
                
                if sim_sum > 0:
                    recommendations[item] = score / sim_sum
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recs[:top_k]]

class MatrixFactorization:
    """矩阵分解推荐算法"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, user_item_matrix):
        """训练矩阵分解模型"""
        print("🔍 训练矩阵分解模型...")
        
        n_users, n_items = user_item_matrix.shape
        
        # 初始化用户和物品因子矩阵
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # 获取非零评分的位置
        user_indices, item_indices = np.nonzero(user_item_matrix.values)
        ratings = user_item_matrix.values[user_indices, item_indices]
        
        # 训练过程
        losses = []
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            
            for i, (user_idx, item_idx, rating) in enumerate(zip(user_indices, item_indices, ratings)):
                # 预测评分
                prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                error = rating - prediction
                
                # 更新因子
                user_factor = self.user_factors[user_idx].copy()
                item_factor = self.item_factors[item_idx].copy()
                
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factor - self.regularization * user_factor
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factor - self.regularization * item_factor
                )
                
                epoch_loss += error ** 2
            
            losses.append(epoch_loss / len(ratings))
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss = {losses[-1]:.4f}")
        
        print(f"✅ 矩阵分解训练完成，最终损失: {losses[-1]:.4f}")
        return losses
    
    def predict(self, user_idx, item_idx):
        """预测用户对物品的评分"""
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
    
    def recommend(self, user_idx, user_item_matrix, top_k=10):
        """为用户推荐物品"""
        user_items = user_item_matrix.iloc[user_idx]
        interacted_items = set(user_items[user_items > 0].index)
        
        recommendations = {}
        for item_idx, item_id in enumerate(user_item_matrix.columns):
            if item_id not in interacted_items:
                score = self.predict(user_idx, item_idx)
                recommendations[item_id] = score
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recs[:top_k]]

class NeuralCollaborativeFiltering(nn.Module):
    """神经协同过滤模型"""
    
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[128, 64]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP层
        mlp_input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                self.mlp_layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            else:
                self.mlp_layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.2))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, item_ids):
        """前向传播"""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接嵌入向量
        mlp_input = torch.cat([user_emb, item_emb], dim=1)
        
        # MLP前向传播
        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)
        
        output = self.output_layer(x)
        return output.squeeze()

class DeepRecommenderSystem:
    """深度学习推荐系统"""
    
    def __init__(self, embedding_dim=50, hidden_dims=[128, 64], learning_rate=0.001):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.model = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data):
        """准备训练数据"""
        print("🔍 准备深度学习训练数据...")
        
        # 创建用户和物品的索引映射
        unique_users = data['user_id'].unique()
        unique_items = data['item_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # 创建交互数据
        interaction_data = data.groupby(['user_id', 'item_id']).size().reset_index(name='rating')
        
        # 转换为索引
        interaction_data['user_idx'] = interaction_data['user_id'].map(self.user_to_idx)
        interaction_data['item_idx'] = interaction_data['item_id'].map(self.item_to_idx)
        
        # 标准化评分
        max_rating = interaction_data['rating'].max()
        interaction_data['rating'] = interaction_data['rating'] / max_rating
        
        print(f"✅ 数据准备完成: {len(interaction_data)} 条交互记录")
        return interaction_data
    
    def train(self, interaction_data, epochs=50, batch_size=1024):
        """训练深度学习模型"""
        print("🔍 训练神经协同过滤模型...")
        
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        # 初始化模型
        self.model = NeuralCollaborativeFiltering(
            n_users, n_items, self.embedding_dim, self.hidden_dims
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 准备数据加载器
        dataset = RecommenderDataset(interaction_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                optimizer.zero_grad()
                
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        print(f"✅ 神经网络训练完成，最终损失: {losses[-1]:.4f}")
        return losses
    
    def recommend(self, user_id, interaction_data, top_k=10):
        """为用户推荐物品"""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # 获取用户已交互的物品
        user_interactions = interaction_data[interaction_data['user_id'] == user_id]
        interacted_items = set(user_interactions['item_id'])
        
        # 预测所有未交互物品的评分
        recommendations = {}
        self.model.eval()
        
        with torch.no_grad():
            for item_id, item_idx in self.item_to_idx.items():
                if item_id not in interacted_items:
                    user_tensor = torch.tensor([user_idx]).to(self.device)
                    item_tensor = torch.tensor([item_idx]).to(self.device)
                    
                    score = self.model(user_tensor, item_tensor).item()
                    recommendations[item_id] = score
        
        # 返回top-k推荐
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recs[:top_k]]

class RecommenderDataset(Dataset):
    """推荐系统数据集"""
    
    def __init__(self, interaction_data):
        self.user_indices = torch.tensor(interaction_data['user_idx'].values, dtype=torch.long)
        self.item_indices = torch.tensor(interaction_data['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(interaction_data['rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.user_indices)
    
    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]

class MultiAlgorithmRecommender:
    """多算法推荐系统集成"""
    
    def __init__(self):
        self.algorithms = {}
        self.evaluation_results = {}
        
    def add_algorithm(self, name, algorithm):
        """添加推荐算法"""
        self.algorithms[name] = algorithm
        print(f"✅ 已添加算法: {name}")
    
    def train_all(self, data):
        """训练所有算法"""
        print("🚀 开始训练所有推荐算法...")
        
        for name, algorithm in self.algorithms.items():
            print(f"\n📈 训练算法: {name}")
            
            if isinstance(algorithm, CollaborativeFiltering):
                algorithm.prepare_data(data)
                algorithm.calculate_similarity()
                
            elif isinstance(algorithm, MatrixFactorization):
                # 准备用户-物品矩阵
                interaction_counts = data.groupby(['user_id', 'item_id']).size().reset_index(name='rating')
                user_item_matrix = interaction_counts.pivot(
                    index='user_id', columns='item_id', values='rating'
                ).fillna(0)
                algorithm.fit(user_item_matrix)
                
            elif isinstance(algorithm, DeepRecommenderSystem):
                interaction_data = algorithm.prepare_data(data)
                algorithm.train(interaction_data)
        
        print("\n🎉 所有算法训练完成!")
    
    def evaluate_algorithms(self, data, test_users=None, top_k=10):
        """评估所有算法"""
        print("📊 开始评估推荐算法...")
        
        if test_users is None:
            # 随机选择测试用户
            all_users = data['user_id'].unique()
            test_users = np.random.choice(all_users, size=min(100, len(all_users)), replace=False)
        
        for name, algorithm in self.algorithms.items():
            print(f"\n📊 评估算法: {name}")
            
            recommendations_count = 0
            total_users = len(test_users)
            
            for user_id in test_users:
                try:
                    if isinstance(algorithm, CollaborativeFiltering):
                        recs = algorithm.recommend(user_id, top_k)
                    elif isinstance(algorithm, MatrixFactorization):
                        # 需要准备矩阵和用户索引
                        interaction_counts = data.groupby(['user_id', 'item_id']).size().reset_index(name='rating')
                        user_item_matrix = interaction_counts.pivot(
                            index='user_id', columns='item_id', values='rating'
                        ).fillna(0)
                        if user_id in user_item_matrix.index:
                            user_idx = user_item_matrix.index.get_loc(user_id)
                            recs = algorithm.recommend(user_idx, user_item_matrix, top_k)
                        else:
                            recs = []
                    elif isinstance(algorithm, DeepRecommenderSystem):
                        interaction_data = algorithm.prepare_data(data)
                        recs = algorithm.recommend(user_id, interaction_data, top_k)
                    
                    if len(recs) > 0:
                        recommendations_count += 1
                        
                except Exception as e:
                    continue
            
            coverage = recommendations_count / total_users
            self.evaluation_results[name] = {
                'coverage': coverage,
                'avg_recommendations': recommendations_count
            }
            
            print(f"  推荐覆盖率: {coverage:.2%}")
            print(f"  成功推荐用户数: {recommendations_count}/{total_users}")
    
    def generate_recommendations(self, user_id, top_k=10):
        """为指定用户生成多算法推荐结果"""
        results = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                if isinstance(algorithm, CollaborativeFiltering):
                    recs = algorithm.recommend(user_id, top_k)
                elif isinstance(algorithm, DeepRecommenderSystem):
                    # 需要重新准备数据，这里简化处理
                    recs = []
                else:
                    recs = []
                
                results[name] = recs
            except Exception as e:
                results[name] = []
        
        return results
    
    def visualize_performance(self):
        """可视化算法性能比较"""
        if not self.evaluation_results:
            print("❌ 请先运行算法评估")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('推荐算法性能比较', fontsize=16, fontweight='bold')
        
        algorithms = list(self.evaluation_results.keys())
        coverage_scores = [self.evaluation_results[alg]['coverage'] for alg in algorithms]
        recommendation_counts = [self.evaluation_results[alg]['avg_recommendations'] for alg in algorithms]
        
        # 推荐覆盖率比较
        axes[0].bar(algorithms, coverage_scores, color='skyblue')
        axes[0].set_title('推荐覆盖率比较')
        axes[0].set_ylabel('覆盖率')
        axes[0].set_ylim(0, 1)
        
        # 成功推荐数比较
        axes[1].bar(algorithms, recommendation_counts, color='lightgreen')
        axes[1].set_title('成功推荐用户数')
        axes[1].set_ylabel('用户数')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    """主函数示例"""
    print("🚀 多算法推荐系统")
    print("=" * 50)
    
    print("📖 使用示例:")
    print("""
    # 1. 初始化多算法推荐器
    multi_recommender = MultiAlgorithmRecommender()
    
    # 2. 添加不同算法
    multi_recommender.add_algorithm('用户协同过滤', CollaborativeFiltering('user_based'))
    multi_recommender.add_algorithm('物品协同过滤', CollaborativeFiltering('item_based'))
    multi_recommender.add_algorithm('矩阵分解', MatrixFactorization())
    multi_recommender.add_algorithm('神经协同过滤', DeepRecommenderSystem())
    
    # 3. 训练所有算法
    multi_recommender.train_all(data)
    
    # 4. 评估算法性能
    multi_recommender.evaluate_algorithms(data)
    
    # 5. 可视化性能比较
    multi_recommender.visualize_performance()
    
    # 6. 为特定用户生成推荐
    recommendations = multi_recommender.generate_recommendations(user_id='12345')
    """)

if __name__ == "__main__":
    main() 