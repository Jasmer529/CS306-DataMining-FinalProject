"""
协同过滤推荐算法实现
包含基于用户和基于物品的协同过滤
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def get_similar_users(self, user_id, top_k=10):
        """获取相似用户"""
        if self.method != 'user_based':
            print("❌ 该功能仅适用于基于用户的协同过滤")
            return []
            
        if user_id not in self.similarity_df.index:
            return []
            
        user_similarities = self.similarity_df.loc[user_id].drop(user_id)
        similar_users = user_similarities.nlargest(top_k)
        
        return [(user, similarity) for user, similarity in similar_users.items()]
    
    def get_similar_items(self, item_id, top_k=10):
        """获取相似物品"""
        if self.method != 'item_based':
            print("❌ 该功能仅适用于基于物品的协同过滤")
            return []
            
        if item_id not in self.similarity_df.index:
            return []
            
        item_similarities = self.similarity_df.loc[item_id].drop(item_id)
        similar_items = item_similarities.nlargest(top_k)
        
        return [(item, similarity) for item, similarity in similar_items.items()]
    
    def visualize_similarity_matrix(self, sample_size=50):
        """可视化相似度矩阵"""
        if self.similarity_matrix is None:
            print("❌ 请先计算相似度矩阵")
            return
            
        # 随机采样显示
        if len(self.similarity_df) > sample_size:
            sample_indices = np.random.choice(len(self.similarity_df), sample_size, replace=False)
            sample_matrix = self.similarity_df.iloc[sample_indices, sample_indices]
        else:
            sample_matrix = self.similarity_df
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(sample_matrix, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.1, cbar_kws={"shrink": .8})
        plt.title(f'{self.method}相似度矩阵热力图 (样本大小: {len(sample_matrix)})')
        plt.tight_layout()
        plt.show()

def demo_collaborative_filtering():
    """协同过滤演示函数"""
    print("🚀 协同过滤推荐算法演示")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    n_users, n_items = 1000, 500
    
    # 生成模拟用户行为数据
    data_list = []
    for _ in range(10000):
        user_id = np.random.randint(1, n_users + 1)
        item_id = np.random.randint(1, n_items + 1)
        behavior_type = np.random.choice(['pv', 'cart', 'fav', 'buy'], 
                                       p=[0.7, 0.15, 0.1, 0.05])
        data_list.append({
            'user_id': user_id,
            'item_id': item_id,
            'behavior_type': behavior_type
        })
    
    demo_data = pd.DataFrame(data_list)
    
    print(f"📊 示例数据: {len(demo_data)} 条行为记录")
    print(f"   用户数: {demo_data['user_id'].nunique()}")
    print(f"   物品数: {demo_data['item_id'].nunique()}")
    
    # 1. 基于用户的协同过滤
    print("\n🔍 测试基于用户的协同过滤...")
    user_cf = CollaborativeFiltering('user_based')
    user_cf.prepare_data(demo_data)
    user_cf.calculate_similarity()
    
    # 为用户推荐
    test_user = demo_data['user_id'].iloc[0]
    recommendations = user_cf.recommend(test_user, top_k=5)
    print(f"为用户 {test_user} 推荐的物品: {recommendations}")
    
    # 获取相似用户
    similar_users = user_cf.get_similar_users(test_user, top_k=3)
    print(f"与用户 {test_user} 最相似的用户: {similar_users}")
    
    # 2. 基于物品的协同过滤
    print("\n🔍 测试基于物品的协同过滤...")
    item_cf = CollaborativeFiltering('item_based')
    item_cf.prepare_data(demo_data)
    item_cf.calculate_similarity()
    
    # 为用户推荐
    recommendations = item_cf.recommend(test_user, top_k=5)
    print(f"为用户 {test_user} 推荐的物品: {recommendations}")
    
    # 获取相似物品
    test_item = demo_data['item_id'].iloc[0]
    similar_items = item_cf.get_similar_items(test_item, top_k=3)
    print(f"与物品 {test_item} 最相似的物品: {similar_items}")
    
    print("\n✅ 协同过滤演示完成!")

if __name__ == "__main__":
    demo_collaborative_filtering() 