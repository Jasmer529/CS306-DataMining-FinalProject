"""
增强版电商用户行为推荐系统
集成所有高级功能：多维度评估、图神经网络、强化学习等
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 导入所有推荐模块
from user_analysis_enhancement import UserAnalysisEnhancer
from collaborative_filtering import CollaborativeFiltering
from transformer_recommender import TransformerRecommender
from advanced_evaluation import AdvancedRecommenderEvaluator
from graph_neural_recommender import GNNRecommenderSystem
from reinforcement_learning_recommender import RLRecommenderSystem

warnings.filterwarnings('ignore')

class EnhancedECommerceRecommenderSystem:
    """增强版电商推荐系统主类"""
    
    def __init__(self, data_path=None):
        """
        初始化增强推荐系统
        :param data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # 分析模块
        self.user_analyzer = None
        self.evaluator = None
        
        # 推荐算法
        self.algorithms = {}
        self.evaluation_results = {}
        
        print("🚀 增强版电商用户行为推荐系统")
        print("=" * 70)
        print("🔥 集成前沿技术:")
        print("   📊 多维度评估 (Precision@K, NDCG, Diversity, Novelty)")
        print("   🕸️  图神经网络推荐 (GCN)")
        print("   🎯 强化学习推荐 (DQN)")
        print("   🤖 Transformer序列推荐")
        print("   👥 协同过滤算法")
        print("=" * 70)
        
    def load_data(self, data_path=None, sample_size=None):
        """
        加载数据
        :param data_path: 数据文件路径
        :param sample_size: 采样大小（用于快速测试）
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            print("❌ 请指定数据文件路径")
            return False
            
        try:
            print(f"📁 正在加载数据: {self.data_path}")
            
            # 读取数据
            if sample_size:
                print(f"📊 采样 {sample_size} 条记录进行快速测试...")
                self.data = pd.read_csv(self.data_path, nrows=sample_size)
            else:
                self.data = pd.read_csv(self.data_path)
            
            # 数据预处理
            self._preprocess_data()
            
            # 数据分割
            self._split_data()
            
            # 初始化评估器
            self.evaluator = AdvancedRecommenderEvaluator(self.train_data)
            
            print(f"✅ 数据加载和预处理完成!")
            self._print_data_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def _preprocess_data(self):
        """数据预处理"""
        # 时间戳处理
        if 'datetime' not in self.data.columns and 'timestamp' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='s')
        elif 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        
        # 添加时间特征
        if 'datetime' in self.data.columns:
            self.data['date'] = self.data['datetime'].dt.date
            self.data['hour'] = self.data['datetime'].dt.hour
            self.data['day_of_week'] = self.data['datetime'].dt.day_of_week
        
        # 数据清洗
        self.data = self.data.dropna(subset=['user_id', 'item_id', 'behavior_type'])
        
        # 过滤低频用户和商品
        user_counts = self.data['user_id'].value_counts()
        item_counts = self.data['item_id'].value_counts()
        
        # 保留至少有5次交互的用户和商品
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        
        self.data = self.data[
            (self.data['user_id'].isin(valid_users)) & 
            (self.data['item_id'].isin(valid_items))
        ]
        
        print(f"📊 数据清洗完成，保留 {len(self.data)} 条记录")
    
    def _split_data(self, test_ratio=0.2):
        """分割训练和测试数据"""
        print(f"🔀 分割数据 (测试集比例: {test_ratio})")
        
        # 时间分割
        self.data = self.data.sort_values('datetime')
        split_point = int(len(self.data) * (1 - test_ratio))
        
        self.train_data = self.data.iloc[:split_point].copy()
        self.test_data = self.data.iloc[split_point:].copy()
        
        print(f"   训练集: {len(self.train_data)} 条记录")
        print(f"   测试集: {len(self.test_data)} 条记录")
    
    def _print_data_summary(self):
        """打印数据摘要"""
        print(f"\n📋 数据摘要:")
        print(f"   📊 总记录数: {len(self.data):,}")
        print(f"   👥 用户数量: {self.data['user_id'].nunique():,}")
        print(f"   🛍️ 商品数量: {self.data['item_id'].nunique():,}")
        
        if 'category_id' in self.data.columns:
            print(f"   🏷️ 品类数量: {self.data['category_id'].nunique():,}")
            
        behavior_counts = self.data['behavior_type'].value_counts()
        print(f"   🎯 行为分布:")
        for behavior, count in behavior_counts.items():
            percentage = count / len(self.data) * 100
            print(f"      {behavior}: {count:,} ({percentage:.1f}%)")
    
    def analyze_users(self):
        """用户行为分析"""
        if self.train_data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n" + "="*60)
        print("👥 用户行为深度分析")
        print("="*60)
        
        # 初始化用户分析器
        self.user_analyzer = UserAnalysisEnhancer(self.train_data)
        
        # 执行分析
        print("\n🔍 进行RFM分析...")
        rfm_data = self.user_analyzer.calculate_rfm_features()
        rfm_segments = self.user_analyzer.rfm_segmentation()
        
        print("\n🔍 计算高级用户特征...")
        user_features = self.user_analyzer.calculate_advanced_features()
        
        print("\n🔍 进行K-means聚类...")
        clusters = self.user_analyzer.kmeans_clustering()
        
        print("\n🔍 生成用户画像...")
        profiles = self.user_analyzer.generate_user_profiles()
        
        # 保存结果
        print("\n💾 保存分析结果...")
        self.user_analyzer.save_analysis_results('enhanced_analysis_results')
        
        print("✅ 用户分析完成!")
        return True
    
    def train_all_algorithms(self):
        """训练所有推荐算法"""
        if self.train_data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n" + "="*60)
        print("🤖 训练所有推荐算法")
        print("="*60)
        
        # 1. 协同过滤算法
        print("\n🔍 训练协同过滤算法...")
        self._train_collaborative_filtering()
        
        # 2. Transformer序列推荐
        print("\n🔍 训练Transformer序列推荐...")
        self._train_transformer_recommender()
        
        # 3. 图神经网络推荐
        print("\n🔍 训练图神经网络推荐...")
        self._train_gnn_recommender()
        
        # 4. 强化学习推荐
        print("\n🔍 训练强化学习推荐...")
        self._train_rl_recommender()
        
        print(f"\n✅ 所有算法训练完成! 共训练 {len(self.algorithms)} 个算法")
        return True
    
    def _train_collaborative_filtering(self):
        """训练协同过滤算法"""
        try:
            # 基于用户的协同过滤
            user_cf = CollaborativeFiltering('user_based')
            user_cf.prepare_data(self.train_data)
            user_cf.calculate_similarity()
            self.algorithms['UserCF'] = user_cf
            print("   ✅ 基于用户的协同过滤训练完成")
            
            # 基于物品的协同过滤
            item_cf = CollaborativeFiltering('item_based')
            item_cf.prepare_data(self.train_data)
            item_cf.calculate_similarity()
            self.algorithms['ItemCF'] = item_cf
            print("   ✅ 基于物品的协同过滤训练完成")
            
        except Exception as e:
            print(f"   ⚠️ 协同过滤训练失败: {e}")
    
    def _train_transformer_recommender(self):
        """训练Transformer推荐模型"""
        try:
            transformer_rec = TransformerRecommender(
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                max_seq_len=50
            )
            
            losses = transformer_rec.train(self.train_data, epochs=20, batch_size=64)
            self.algorithms['Transformer'] = transformer_rec
            print("   ✅ Transformer序列推荐训练完成")
            
        except Exception as e:
            print(f"   ⚠️ Transformer训练失败: {e}")
    
    def _train_gnn_recommender(self):
        """训练图神经网络推荐模型"""
        try:
            gnn_rec = GNNRecommenderSystem(
                embedding_dim=32,
                hidden_dim=16,
                num_layers=2,
                learning_rate=0.001
            )
            
            gnn_rec.setup(self.train_data)
            losses, val_losses = gnn_rec.train(
                self.train_data, 
                epochs=30, 
                batch_size=512
            )
            self.algorithms['GNN'] = gnn_rec
            print("   ✅ 图神经网络推荐训练完成")
            
        except Exception as e:
            print(f"   ⚠️ GNN训练失败: {e}")
    
    def _train_rl_recommender(self):
        """训练强化学习推荐模型"""
        try:
            rl_rec = RLRecommenderSystem(
                state_dim=50,
                hidden_dim=128,
                learning_rate=0.001
            )
            
            rl_rec.setup(self.train_data)
            episode_rewards, episode_losses = rl_rec.train(
                episodes=200,
                max_steps_per_episode=10
            )
            self.algorithms['RL'] = rl_rec
            print("   ✅ 强化学习推荐训练完成")
            
        except Exception as e:
            print(f"   ⚠️ 强化学习训练失败: {e}")
    
    def evaluate_all_algorithms(self, top_k=10):
        """评估所有算法性能"""
        if not self.algorithms:
            print("❌ 请先训练算法")
            return
            
        print("\n" + "="*60)
        print("📊 全面算法性能评估")
        print("="*60)
        
        # 使用高级评估器进行评估
        self.evaluation_results = self.evaluator.compare_algorithms(
            self.algorithms, 
            self.test_data, 
            k=top_k
        )
        
        # 生成评估报告
        self.evaluator.generate_evaluation_report(
            self.evaluation_results, 
            'enhanced_evaluation_report.md'
        )
        
        # 可视化评估结果
        self._visualize_evaluation_results()
        
        print("✅ 算法评估完成!")
        return self.evaluation_results
    
    def _visualize_evaluation_results(self):
        """可视化评估结果"""
        if self.evaluation_results is None or self.evaluation_results.empty:
            return
        
        print("\n📊 生成评估结果可视化...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('推荐算法性能对比', fontsize=16, fontweight='bold')
        
        # 1. 准确性指标对比
        metrics = ['precision_mean', 'recall_mean', 'ndcg_mean']
        ax1 = axes[0, 0]
        self.evaluation_results[metrics].plot(kind='bar', ax=ax1)
        ax1.set_title('准确性指标对比')
        ax1.set_ylabel('指标值')
        ax1.legend(['Precision@10', 'Recall@10', 'NDCG@10'])
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 多样性和覆盖率
        ax2 = axes[0, 1]
        diversity_coverage = self.evaluation_results[['diversity_mean', 'coverage']]
        diversity_coverage.plot(kind='bar', ax=ax2)
        ax2.set_title('多样性和覆盖率')
        ax2.set_ylabel('指标值')
        ax2.legend(['多样性', '覆盖率'])
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. F1分数雷达图
        ax3 = axes[1, 0]
        f1_scores = self.evaluation_results['f1_score'].sort_values(ascending=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(f1_scores)))
        bars = ax3.barh(range(len(f1_scores)), f1_scores.values, color=colors)
        ax3.set_yticks(range(len(f1_scores)))
        ax3.set_yticklabels(f1_scores.index)
        ax3.set_xlabel('F1分数')
        ax3.set_title('F1分数排名')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, f1_scores.values)):
            ax3.text(value + 0.001, i, f'{value:.3f}', 
                    va='center', ha='left', fontsize=10)
        
        # 4. 综合性能热力图
        ax4 = axes[1, 1]
        heatmap_data = self.evaluation_results[
            ['precision_mean', 'recall_mean', 'ndcg_mean', 'diversity_mean', 'coverage']
        ].T
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('综合性能热力图')
        ax4.set_xlabel('算法')
        ax4.set_ylabel('评估指标')
        
        plt.tight_layout()
        plt.savefig('enhanced_algorithm_evaluation.png', dpi=300, bbox_inches='tight')
        print("   💾 评估图表已保存: enhanced_algorithm_evaluation.png")
        
        # 显示图表
        plt.show()
    
    def generate_recommendations(self, user_id, algorithm='best', top_k=10):
        """为用户生成推荐"""
        if not self.algorithms:
            print("❌ 请先训练算法")
            return []
        
        # 选择算法
        if algorithm == 'best' and hasattr(self, 'evaluation_results') and not self.evaluation_results.empty:
            # 使用F1分数最高的算法
            best_algorithm_name = self.evaluation_results['f1_score'].idxmax()
            selected_algorithm = self.algorithms[best_algorithm_name]
            print(f"🎯 使用最佳算法: {best_algorithm_name}")
        elif algorithm in self.algorithms:
            selected_algorithm = self.algorithms[algorithm]
            print(f"🎯 使用指定算法: {algorithm}")
        else:
            # 默认使用第一个可用算法
            algorithm_name = list(self.algorithms.keys())[0]
            selected_algorithm = self.algorithms[algorithm_name]
            print(f"🎯 使用默认算法: {algorithm_name}")
        
        # 生成推荐
        try:
            if hasattr(selected_algorithm, 'recommend'):
                recommendations = selected_algorithm.recommend(user_id, top_k=top_k)
            elif hasattr(selected_algorithm, 'recommend_for_user'):
                recommendations = selected_algorithm.recommend_for_user(user_id, top_k=top_k)
            else:
                print("❌ 算法不支持推荐功能")
                return []
            
            print(f"✅ 为用户 {user_id} 生成 {len(recommendations)} 个推荐")
            return recommendations
            
        except Exception as e:
            print(f"❌ 推荐生成失败: {e}")
            return []
    
    def run_complete_analysis(self, sample_size=None):
        """运行完整分析流程"""
        print("\n🚀 开始完整的推荐系统分析流程")
        print("=" * 70)
        
        # 1. 加载数据
        if not self.load_data(sample_size=sample_size):
            return False
        
        # 2. 用户分析
        if not self.analyze_users():
            return False
        
        # 3. 训练所有算法
        if not self.train_all_algorithms():
            return False
        
        # 4. 评估算法性能
        if not self.evaluate_all_algorithms():
            return False
        
        # 5. 生成总结报告
        self._generate_final_report()
        
        print("\n🎉 完整分析流程执行完成!")
        print("📄 查看以下输出文件:")
        print("   📊 enhanced_analysis_results/ - 用户分析结果")
        print("   📋 enhanced_evaluation_report.md - 算法评估报告")
        print("   📈 enhanced_algorithm_evaluation.png - 性能对比图")
        print("   📝 enhanced_final_report.md - 完整分析报告")
        
        return True
    
    def _generate_final_report(self):
        """生成最终报告"""
        with open('enhanced_final_report.md', 'w', encoding='utf-8') as f:
            f.write("# 增强版电商推荐系统分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据概况
            f.write("## 1. 数据概况\n\n")
            f.write(f"- **总记录数**: {len(self.data):,}\n")
            f.write(f"- **用户数量**: {self.data['user_id'].nunique():,}\n")
            f.write(f"- **商品数量**: {self.data['item_id'].nunique():,}\n")
            
            if 'category_id' in self.data.columns:
                f.write(f"- **品类数量**: {self.data['category_id'].nunique():,}\n")
            
            f.write("\n### 行为分布\n\n")
            behavior_counts = self.data['behavior_type'].value_counts()
            for behavior, count in behavior_counts.items():
                percentage = count / len(self.data) * 100
                f.write(f"- **{behavior}**: {count:,} ({percentage:.1f}%)\n")
            
            # 算法性能
            f.write("\n## 2. 算法性能对比\n\n")
            if hasattr(self, 'evaluation_results') and not self.evaluation_results.empty:
                f.write("### 主要性能指标\n\n")
                f.write(self.evaluation_results[
                    ['precision_mean', 'recall_mean', 'ndcg_mean', 'diversity_mean', 'coverage', 'f1_score']
                ].round(4).to_markdown())
                
                best_algorithm = self.evaluation_results['f1_score'].idxmax()
                f.write(f"\n**最佳算法**: {best_algorithm}\n")
                f.write(f"**F1分数**: {self.evaluation_results.loc[best_algorithm, 'f1_score']:.4f}\n")
            
            # 技术特点
            f.write("\n## 3. 技术特点\n\n")
            f.write("本推荐系统集成了以下前沿技术:\n\n")
            f.write("### 🤖 深度学习技术\n")
            f.write("- **Transformer序列推荐**: 使用自注意力机制捕获用户行为序列模式\n")
            f.write("- **图神经网络**: 基于用户-商品二部图进行协同过滤\n")
            f.write("- **强化学习**: 使用DQN学习推荐策略\n\n")
            
            f.write("### 📊 评估体系\n")
            f.write("- **多维度评估**: Precision@K, Recall@K, NDCG@K\n")
            f.write("- **多样性评估**: 推荐列表多样性和新颖性\n")
            f.write("- **覆盖率评估**: 商品覆盖和长尾推荐能力\n\n")
            
            f.write("### 👥 用户分析\n")
            f.write("- **RFM分析**: 用户价值分群\n")
            f.write("- **行为模式挖掘**: 转化漏斗和偏好分析\n")
            f.write("- **用户画像**: 多维度标签体系\n\n")
            
            # 应用建议
            f.write("## 4. 应用建议\n\n")
            f.write("### 算法选择建议\n")
            if hasattr(self, 'evaluation_results') and not self.evaluation_results.empty:
                for algorithm in self.evaluation_results.index:
                    precision = self.evaluation_results.loc[algorithm, 'precision_mean']
                    diversity = self.evaluation_results.loc[algorithm, 'diversity_mean']
                    
                    if precision > 0.1:
                        if diversity > 0.7:
                            scenario = "适合首页推荐和个性化发现"
                        else:
                            scenario = "适合购买转化和精准营销"
                    else:
                        scenario = "适合冷启动和探索性推荐"
                    
                    f.write(f"- **{algorithm}**: {scenario}\n")
            
            f.write("\n### 业务价值\n")
            f.write("- **个性化推荐**: 提升用户体验和转化率\n")
            f.write("- **用户运营**: 精准用户分群和生命周期管理\n")
            f.write("- **商品运营**: 新品推广和库存优化\n")
            f.write("- **数据驱动**: 基于数据的决策支持\n")
        
        print("📝 最终报告已生成: enhanced_final_report.md")

def demo_enhanced_system():
    """演示增强版推荐系统"""
    print("🎬 增强版推荐系统演示")
    print("=" * 50)
    
    # 初始化系统
    system = EnhancedECommerceRecommenderSystem()
    
    # 选择数据文件
    data_file = input("请输入数据文件路径 (回车使用默认 UserBehavior.csv): ").strip()
    if not data_file:
        data_file = "UserBehavior.csv"
    
    # 选择采样大小
    sample_input = input("请输入采样大小 (回车使用全量数据，建议测试时使用 100000): ").strip()
    sample_size = None
    if sample_input.isdigit():
        sample_size = int(sample_input)
    
    # 运行完整分析
    success = system.run_complete_analysis(sample_size=sample_size)
    
    if success:
        print("\n🎯 演示推荐生成:")
        
        # 获取一些用户ID进行推荐演示
        sample_users = system.data['user_id'].unique()[:5]
        
        for user_id in sample_users:
            print(f"\n👤 为用户 {user_id} 生成推荐:")
            recommendations = system.generate_recommendations(user_id, top_k=5)
            
            if recommendations:
                for i, item_id in enumerate(recommendations, 1):
                    print(f"   {i}. 商品 {item_id}")
            else:
                print("   暂无推荐")
    
    return system

def main():
    """主程序入口"""
    print("🛒 增强版电商用户行为推荐系统")
    print("=" * 50)
    print("请选择运行模式:")
    print("1. 快速演示 (采样数据)")
    print("2. 完整分析 (全量数据)")
    print("3. 自定义分析")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        system = EnhancedECommerceRecommenderSystem()
        system.run_complete_analysis(sample_size=50000)  # 5万条记录快速测试
        
    elif choice == "2":
        system = EnhancedECommerceRecommenderSystem()
        system.run_complete_analysis()  # 全量数据分析
        
    elif choice == "3":
        demo_enhanced_system()
        
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main() 