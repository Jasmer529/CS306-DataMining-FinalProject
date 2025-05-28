"""
用户行为分析增强模块
包含用户分群、RFM分析、特征工程等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UserAnalysisEnhancer:
    """用户行为深度分析类"""
    
    def __init__(self, data):
        """
        初始化
        :param data: 用户行为数据DataFrame
        """
        self.data = data.copy()
        self.user_features = None
        self.rfm_data = None
        self.user_clusters = None
        
    def calculate_rfm_features(self):
        """计算RFM特征"""
        print("🔍 计算RFM特征...")
        
        # 获取分析时间点（数据最后一天的下一天）
        max_date = self.data['datetime'].max()
        analysis_date = max_date + timedelta(days=1)
        
        # 计算RFM指标
        rfm_data = self.data.groupby('user_id').agg({
            'datetime': lambda x: (analysis_date - x.max()).days,  # Recency
            'behavior_type': 'count',  # Frequency
            'item_id': 'nunique'  # 浏览商品数量
        }).rename(columns={
            'datetime': 'recency',
            'behavior_type': 'frequency', 
            'item_id': 'item_variety'
        })
        
        # 计算购买金额（货币价值）- 这里用购买次数代替
        purchase_data = self.data[self.data['behavior_type'] == 'buy'].groupby('user_id').agg({
            'behavior_type': 'count'
        }).rename(columns={'behavior_type': 'monetary'})
        
        # 合并RFM数据
        rfm_data = rfm_data.join(purchase_data, how='left')
        rfm_data['monetary'] = rfm_data['monetary'].fillna(0)
        
        self.rfm_data = rfm_data
        print(f"✅ RFM特征计算完成，用户数量: {len(rfm_data)}")
        return rfm_data
    
    def calculate_advanced_features(self):
        """计算高级用户特征"""
        print("🔍 计算高级用户特征...")
        
        features_list = []
        
        for user_id in self.data['user_id'].unique():
            user_data = self.data[self.data['user_id'] == user_id]
            
            # 基础统计特征
            total_actions = len(user_data)
            unique_items = user_data['item_id'].nunique()
            unique_categories = user_data['category_id'].nunique()
            
            # 行为类型分布
            behavior_counts = user_data['behavior_type'].value_counts()
            pv_count = behavior_counts.get('pv', 0)
            cart_count = behavior_counts.get('cart', 0)
            fav_count = behavior_counts.get('fav', 0)
            buy_count = behavior_counts.get('buy', 0)
            
            # 转化率计算
            pv_to_cart_rate = cart_count / pv_count if pv_count > 0 else 0
            pv_to_buy_rate = buy_count / pv_count if pv_count > 0 else 0
            cart_to_buy_rate = buy_count / cart_count if cart_count > 0 else 0
            
            # 时间特征
            time_span = (user_data['datetime'].max() - user_data['datetime'].min()).days
            avg_daily_actions = total_actions / max(time_span, 1)
            
            # 活跃时段分析
            hour_counts = user_data['hour'].value_counts()
            most_active_hour = hour_counts.index[0] if not hour_counts.empty else 0
            
            # 品类集中度 (基尼系数)
            category_dist = user_data['category_id'].value_counts(normalize=True)
            category_concentration = 1 - (category_dist ** 2).sum()
            
            # 会话分析（相邻操作时间间隔超过30分钟认为是新会话）
            user_data_sorted = user_data.sort_values('datetime')
            time_diffs = user_data_sorted['datetime'].diff()
            session_breaks = (time_diffs > timedelta(minutes=30)).sum()
            avg_session_length = total_actions / max(session_breaks + 1, 1)
            
            features = {
                'user_id': user_id,
                'total_actions': total_actions,
                'unique_items': unique_items,
                'unique_categories': unique_categories,
                'pv_count': pv_count,
                'cart_count': cart_count,
                'fav_count': fav_count,
                'buy_count': buy_count,
                'pv_to_cart_rate': pv_to_cart_rate,
                'pv_to_buy_rate': pv_to_buy_rate,
                'cart_to_buy_rate': cart_to_buy_rate,
                'time_span_days': time_span,
                'avg_daily_actions': avg_daily_actions,
                'most_active_hour': most_active_hour,
                'category_concentration': category_concentration,
                'avg_session_length': avg_session_length,
                'session_count': session_breaks + 1
            }
            
            features_list.append(features)
        
        self.user_features = pd.DataFrame(features_list)
        print(f"✅ 高级特征计算完成，特征数量: {len(self.user_features.columns)-1}")
        return self.user_features
    
    def rfm_segmentation(self):
        """基于RFM进行用户分群"""
        if self.rfm_data is None:
            self.calculate_rfm_features()
            
        print("🎯 进行RFM用户分群...")
        
        # RFM打分 (1-5分)
        rfm_scores = self.rfm_data.copy()
        
        # Recency: 越小越好 (最近购买)
        rfm_scores['R_score'] = pd.qcut(rfm_scores['recency'].rank(method='first'), 
                                       5, labels=[5,4,3,2,1])
        
        # Frequency: 越大越好
        rfm_scores['F_score'] = pd.qcut(rfm_scores['frequency'].rank(method='first'), 
                                       5, labels=[1,2,3,4,5])
        
        # Monetary: 越大越好
        rfm_scores['M_score'] = pd.qcut(rfm_scores['monetary'].rank(method='first'), 
                                       5, labels=[1,2,3,4,5])
        
        # 合成RFM得分
        rfm_scores['RFM_score'] = (rfm_scores['R_score'].astype(str) + 
                                  rfm_scores['F_score'].astype(str) + 
                                  rfm_scores['M_score'].astype(str))
        
        # 用户分群规则
        def segment_users(row):
            score = row['RFM_score']
            r, f, m = int(score[0]), int(score[1]), int(score[2])
            
            if r >= 4 and f >= 4 and m >= 4:
                return "冠军用户"
            elif r >= 3 and f >= 3 and m >= 3:
                return "忠诚用户"
            elif r >= 4 and f <= 2:
                return "新用户"
            elif r <= 2 and f >= 3 and m >= 3:
                return "流失风险用户"
            elif r <= 2 and f <= 2:
                return "已流失用户"
            elif f >= 4 and m <= 2:
                return "潜力用户" 
            else:
                return "一般用户"
        
        rfm_scores['segment'] = rfm_scores.apply(segment_users, axis=1)
        
        # 统计各分群
        segment_stats = rfm_scores['segment'].value_counts()
        print("\n📊 用户分群结果:")
        for segment, count in segment_stats.items():
            percentage = count / len(rfm_scores) * 100
            print(f"  {segment}: {count:,} 用户 ({percentage:.1f}%)")
        
        self.rfm_scores = rfm_scores
        return rfm_scores
    
    def kmeans_clustering(self, n_clusters=None):
        """基于用户特征进行K-means聚类"""
        if self.user_features is None:
            self.calculate_advanced_features()
            
        print("🎯 进行K-means用户聚类...")
        
        # 选择聚类特征
        cluster_features = [
            'total_actions', 'unique_items', 'unique_categories',
            'pv_to_cart_rate', 'pv_to_buy_rate', 'avg_daily_actions',
            'category_concentration', 'avg_session_length'
        ]
        
        X = self.user_features[cluster_features].fillna(0)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 确定最优聚类数
        if n_clusters is None:
            silhouette_scores = []
            K_range = range(2, 11)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # 选择最优K值
            optimal_k = K_range[np.argmax(silhouette_scores)]
            print(f"📈 最优聚类数: {optimal_k} (轮廓系数: {max(silhouette_scores):.3f})")
        else:
            optimal_k = n_clusters
        
        # 执行聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 添加聚类结果
        self.user_features['cluster'] = cluster_labels
        
        # 分析各聚类特征
        print("\n📊 聚类结果分析:")
        for cluster_id in range(optimal_k):
            cluster_data = self.user_features[self.user_features['cluster'] == cluster_id]
            size = len(cluster_data)
            avg_actions = cluster_data['total_actions'].mean()
            avg_buy_rate = cluster_data['pv_to_buy_rate'].mean()
            
            print(f"  聚类 {cluster_id}: {size:,} 用户 | "
                  f"平均行为数: {avg_actions:.1f} | "
                  f"平均购买转化率: {avg_buy_rate:.3f}")
        
        self.cluster_model = kmeans
        self.scaler = scaler
        return self.user_features
    
    def visualize_user_segments(self):
        """可视化用户分群结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('用户分群分析可视化', fontsize=16, fontweight='bold')
        
        # 1. RFM分群分布
        if hasattr(self, 'rfm_scores'):
            ax1 = axes[0, 0]
            segment_counts = self.rfm_scores['segment'].value_counts()
            ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
            ax1.set_title('RFM用户分群分布')
        
        # 2. 聚类特征分布
        if 'cluster' in self.user_features.columns:
            ax2 = axes[0, 1]
            self.user_features['cluster'].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title('K-means聚类分布')
            ax2.set_xlabel('聚类ID')
            ax2.set_ylabel('用户数量')
        
        # 3. 用户活跃度分布
        ax3 = axes[1, 0]
        self.user_features['total_actions'].hist(bins=50, ax=ax3, alpha=0.7)
        ax3.set_title('用户活跃度分布')
        ax3.set_xlabel('总行为数量')
        ax3.set_ylabel('用户数量')
        
        # 4. 购买转化率分布
        ax4 = axes[1, 1]
        self.user_features['pv_to_buy_rate'].hist(bins=50, ax=ax4, alpha=0.7)
        ax4.set_title('购买转化率分布')
        ax4.set_xlabel('转化率')
        ax4.set_ylabel('用户数量')
        
        plt.tight_layout()
        plt.show()
    
    def generate_user_profiles(self):
        """生成用户画像报告"""
        print("📋 生成用户画像报告...")
        
        profiles = {}
        
        # RFM分群画像
        if hasattr(self, 'rfm_scores'):
            for segment in self.rfm_scores['segment'].unique():
                segment_data = self.rfm_scores[self.rfm_scores['segment'] == segment]
                
                profile = {
                    '用户数量': len(segment_data),
                    '平均最近性': segment_data['recency'].mean(),
                    '平均频率': segment_data['frequency'].mean(),
                    '平均购买次数': segment_data['monetary'].mean(),
                    '占比': len(segment_data) / len(self.rfm_scores) * 100
                }
                
                profiles[segment] = profile
        
        return profiles
    
    def save_analysis_results(self, output_dir='analysis_results'):
        """保存分析结果"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存用户特征
        if self.user_features is not None:
            self.user_features.to_csv(f'{output_dir}/user_features.csv', index=False)
            print(f"✅ 用户特征已保存到 {output_dir}/user_features.csv")
        
        # 保存RFM分析结果
        if hasattr(self, 'rfm_scores'):
            self.rfm_scores.to_csv(f'{output_dir}/rfm_analysis.csv', index=False)
            print(f"✅ RFM分析结果已保存到 {output_dir}/rfm_analysis.csv")
        
        # 保存用户画像
        profiles = self.generate_user_profiles()
        if profiles:
            profile_df = pd.DataFrame(profiles).T
            profile_df.to_csv(f'{output_dir}/user_profiles.csv')
            print(f"✅ 用户画像已保存到 {output_dir}/user_profiles.csv")

def main():
    """主函数示例"""
    print("🚀 用户行为分析增强模块")
    print("=" * 50)
    
    # 示例用法
    print("📖 使用示例:")
    print("""
    # 1. 加载数据
    data = pd.read_csv('UserBehavior.csv')
    
    # 2. 初始化分析器
    analyzer = UserAnalysisEnhancer(data)
    
    # 3. 计算RFM特征
    rfm_data = analyzer.calculate_rfm_features()
    
    # 4. 计算高级特征
    user_features = analyzer.calculate_advanced_features()
    
    # 5. RFM分群
    rfm_segments = analyzer.rfm_segmentation()
    
    # 6. K-means聚类
    clusters = analyzer.kmeans_clustering()
    
    # 7. 可视化结果
    analyzer.visualize_user_segments()
    
    # 8. 保存分析结果
    analyzer.save_analysis_results()
    """)

if __name__ == "__main__":
    main() 