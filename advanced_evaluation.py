"""
高级推荐系统评估模块
包含多维度评估指标：准确性、多样性、新颖性、覆盖率等
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

class AdvancedRecommenderEvaluator:
    """推荐系统高级评估器"""
    
    def __init__(self, data):
        """
        初始化评估器
        :param data: 原始数据DataFrame
        """
        self.data = data
        self.user_item_matrix = None
        self.item_popularity = None
        self.prepare_evaluation_data()
    
    def prepare_evaluation_data(self):
        """准备评估数据"""
        # 创建用户-商品交互矩阵
        self.user_item_matrix = self.data.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='behavior_type',
            aggfunc='count',
            fill_value=0
        )
        
        # 计算商品流行度
        self.item_popularity = self.data['item_id'].value_counts().to_dict()
        
        print(f"📊 评估数据准备完成:")
        print(f"   用户数量: {len(self.user_item_matrix)}")
        print(f"   商品数量: {len(self.user_item_matrix.columns)}")
    
    def split_data(self, test_ratio=0.2, random_state=42):
        """
        分割数据为训练集和测试集
        :param test_ratio: 测试集比例
        :param random_state: 随机种子
        :return: train_data, test_data
        """
        # 按时间排序
        data_sorted = self.data.sort_values('datetime')
        
        # 为每个用户分割数据
        train_list = []
        test_list = []
        
        for user_id in self.data['user_id'].unique():
            user_data = data_sorted[data_sorted['user_id'] == user_id]
            n_test = max(1, int(len(user_data) * test_ratio))
            
            train_data = user_data.iloc[:-n_test]
            test_data = user_data.iloc[-n_test:]
            
            train_list.append(train_data)
            test_list.append(test_data)
        
        train_data = pd.concat(train_list, ignore_index=True)
        test_data = pd.concat(test_list, ignore_index=True)
        
        return train_data, test_data
    
    def precision_at_k(self, actual, predicted, k=10):
        """计算Precision@K"""
        if not predicted:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant = len(set(predicted_k) & set(actual))
        return relevant / min(k, len(predicted_k))
    
    def recall_at_k(self, actual, predicted, k=10):
        """计算Recall@K"""
        if not actual:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant = len(set(predicted_k) & set(actual))
        return relevant / len(actual)
    
    def ndcg_at_k(self, actual, predicted, k=10):
        """计算NDCG@K (Normalized Discounted Cumulative Gain)"""
        if not predicted:
            return 0.0
        
        predicted_k = predicted[:k]
        
        # 计算DCG
        dcg = 0.0
        for i, item in enumerate(predicted_k):
            if item in actual:
                dcg += 1.0 / math.log2(i + 2)
        
        # 计算IDCG
        idcg = 0.0
        for i in range(min(len(actual), k)):
            idcg += 1.0 / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def diversity_score(self, recommendations, item_features=None):
        """
        计算推荐列表的多样性
        使用项目间的平均距离
        """
        if len(recommendations) < 2:
            return 0.0
        
        if item_features is not None:
            # 基于特征的多样性
            diversity = 0.0
            count = 0
            for i in range(len(recommendations)):
                for j in range(i+1, len(recommendations)):
                    item_i = recommendations[i]
                    item_j = recommendations[j]
                    if item_i in item_features and item_j in item_features:
                        # 计算余弦距离
                        features_i = np.array(item_features[item_i])
                        features_j = np.array(item_features[item_j])
                        similarity = np.dot(features_i, features_j) / (
                            np.linalg.norm(features_i) * np.linalg.norm(features_j)
                        )
                        diversity += 1 - similarity
                        count += 1
            
            return diversity / count if count > 0 else 0.0
        else:
            # 基于品类的多样性
            categories = []
            for item in recommendations:
                item_data = self.data[self.data['item_id'] == item]
                if not item_data.empty and 'category_id' in item_data.columns:
                    categories.append(item_data['category_id'].iloc[0])
            
            if not categories:
                return 0.0
            
            unique_categories = len(set(categories))
            return unique_categories / len(categories)
    
    def novelty_score(self, recommendations):
        """
        计算推荐列表的新颖性
        基于商品流行度的负对数
        """
        if not recommendations:
            return 0.0
        
        total_novelty = 0.0
        total_items = len(self.data)
        
        for item in recommendations:
            if item in self.item_popularity:
                popularity = self.item_popularity[item]
                novelty = -math.log2(popularity / total_items)
                total_novelty += novelty
        
        return total_novelty / len(recommendations)
    
    def coverage_score(self, all_recommendations):
        """
        计算推荐系统的覆盖率
        :param all_recommendations: 所有用户的推荐列表
        """
        recommended_items = set()
        for recommendations in all_recommendations:
            recommended_items.update(recommendations)
        
        total_items = len(self.user_item_matrix.columns)
        return len(recommended_items) / total_items
    
    def evaluate_algorithm(self, algorithm, test_data, algorithm_name="Unknown", k=10):
        """
        全面评估单个算法
        :param algorithm: 推荐算法对象
        :param test_data: 测试数据
        :param algorithm_name: 算法名称
        :param k: Top-K推荐数量
        :return: 评估结果字典
        """
        print(f"\n📊 评估算法: {algorithm_name}")
        
        # 获取测试用户
        test_users = test_data['user_id'].unique()
        
        metrics = {
            'precision': [],
            'recall': [],
            'ndcg': [],
            'diversity': [],
            'novelty': []
        }
        
        all_recommendations = []
        successful_count = 0
        
        for user_id in test_users:
            try:
                # 获取用户的真实交互项目
                actual_items = test_data[test_data['user_id'] == user_id]['item_id'].tolist()
                
                # 生成推荐
                if hasattr(algorithm, 'recommend'):
                    recommendations = algorithm.recommend(user_id, top_k=k)
                elif hasattr(algorithm, 'recommend_for_user'):
                    recommendations = algorithm.recommend_for_user(
                        self.data[self.data['user_id'] != user_id], user_id, top_k=k
                    )
                else:
                    continue
                
                if not recommendations:
                    continue
                
                # 计算指标
                precision = self.precision_at_k(actual_items, recommendations, k)
                recall = self.recall_at_k(actual_items, recommendations, k)
                ndcg = self.ndcg_at_k(actual_items, recommendations, k)
                diversity = self.diversity_score(recommendations)
                novelty = self.novelty_score(recommendations)
                
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['ndcg'].append(ndcg)
                metrics['diversity'].append(diversity)
                metrics['novelty'].append(novelty)
                
                all_recommendations.append(recommendations)
                successful_count += 1
                
            except Exception as e:
                continue
        
        # 计算平均指标
        results = {}
        for metric, values in metrics.items():
            if values:
                results[f'{metric}_mean'] = np.mean(values)
                results[f'{metric}_std'] = np.std(values)
            else:
                results[f'{metric}_mean'] = 0.0
                results[f'{metric}_std'] = 0.0
        
        # 计算覆盖率
        results['coverage'] = self.coverage_score(all_recommendations)
        results['success_rate'] = successful_count / len(test_users)
        
        print(f"   ✅ 成功评估 {successful_count}/{len(test_users)} 个用户")
        print(f"   📈 Precision@{k}: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"   📈 Recall@{k}: {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"   📈 NDCG@{k}: {results['ndcg_mean']:.4f} ± {results['ndcg_std']:.4f}")
        print(f"   🎯 Diversity: {results['diversity_mean']:.4f} ± {results['diversity_std']:.4f}")
        print(f"   🆕 Novelty: {results['novelty_mean']:.4f} ± {results['novelty_std']:.4f}")
        print(f"   📊 Coverage: {results['coverage']:.4f}")
        
        return results
    
    def compare_algorithms(self, algorithms, test_data, k=10):
        """
        比较多个算法的性能
        :param algorithms: 算法字典 {name: algorithm}
        :param test_data: 测试数据
        :param k: Top-K推荐数量
        :return: 比较结果DataFrame
        """
        print("\n" + "="*60)
        print("🔍 开始算法性能比较")
        print("="*60)
        
        comparison_results = {}
        
        for alg_name, algorithm in algorithms.items():
            results = self.evaluate_algorithm(algorithm, test_data, alg_name, k)
            comparison_results[alg_name] = results
        
        # 转换为DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        # 按主要指标排序
        comparison_df['f1_score'] = 2 * (comparison_df['precision_mean'] * comparison_df['recall_mean']) / (
            comparison_df['precision_mean'] + comparison_df['recall_mean'] + 1e-8
        )
        
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        print("\n🏆 算法性能排名:")
        print(comparison_df[['precision_mean', 'recall_mean', 'ndcg_mean', 'diversity_mean', 'coverage', 'f1_score']].round(4))
        
        return comparison_df
    
    def generate_evaluation_report(self, comparison_results, output_path='evaluation_report.md'):
        """生成评估报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 推荐系统算法评估报告\n\n")
            f.write(f"**评估时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 评估指标说明\n\n")
            f.write("- **Precision@K**: 推荐列表中相关项目的比例\n")
            f.write("- **Recall@K**: 用户相关项目中被推荐的比例\n")
            f.write("- **NDCG@K**: 考虑排序质量的评估指标\n")
            f.write("- **Diversity**: 推荐列表的多样性\n")
            f.write("- **Novelty**: 推荐项目的新颖性\n")
            f.write("- **Coverage**: 推荐系统覆盖的商品比例\n\n")
            
            f.write("## 算法性能对比\n\n")
            f.write(comparison_results.round(4).to_markdown())
            
            f.write("\n\n## 性能分析\n\n")
            best_algorithm = comparison_results.index[0]
            f.write(f"**最佳算法**: {best_algorithm}\n\n")
            f.write("**各指标最优算法**:\n")
            
            metrics = ['precision_mean', 'recall_mean', 'ndcg_mean', 'diversity_mean', 'coverage']
            for metric in metrics:
                best_alg = comparison_results[metric].idxmax()
                best_value = comparison_results.loc[best_alg, metric]
                f.write(f"- {metric}: {best_alg} ({best_value:.4f})\n")
        
        print(f"📄 评估报告已保存至: {output_path}") 