"""
电商用户行为推荐系统主程序
整合用户分析、多算法推荐、可视化等功能
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import sys

# 导入自定义模块
from user_analysis_enhancement import UserAnalysisEnhancer
from collaborative_filtering import CollaborativeFiltering
from transformer_recommender import TransformerRecommender

warnings.filterwarnings('ignore')

class ECommerceRecommenderSystem:
    """电商推荐系统主类"""
    
    def __init__(self, data_path=None):
        """
        初始化推荐系统
        :param data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.user_analyzer = None
        self.algorithms = {}
        self.results = {}
        
        print("🛒 电商用户行为推荐系统")
        print("=" * 60)
        
    def load_data(self, data_path=None):
        """加载数据"""
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            print("❌ 请指定数据文件路径")
            return False
            
        try:
            print(f"📁 正在加载数据: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # 数据预处理
            if 'datetime' not in self.data.columns and 'timestamp' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='s')
            elif 'datetime' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            
            # 添加时间特征
            if 'datetime' in self.data.columns:
                self.data['date'] = self.data['datetime'].dt.date
                self.data['hour'] = self.data['datetime'].dt.hour
                self.data['day_of_week'] = self.data['datetime'].dt.day_of_week
            
            print(f"✅ 数据加载成功!")
            print(f"   📊 总记录数: {len(self.data):,}")
            print(f"   👥 用户数量: {self.data['user_id'].nunique():,}")
            print(f"   🛍️ 商品数量: {self.data['item_id'].nunique():,}")
            
            if 'category_id' in self.data.columns:
                print(f"   🏷️ 品类数量: {self.data['category_id'].nunique():,}")
                
            behavior_counts = self.data['behavior_type'].value_counts()
            print(f"   🎯 行为分布: {dict(behavior_counts)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def analyze_users(self):
        """用户行为分析"""
        if self.data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n" + "="*50)
        print("👥 开始用户行为分析")
        print("="*50)
        
        # 初始化用户分析器
        self.user_analyzer = UserAnalysisEnhancer(self.data)
        
        # 1. RFM分析
        print("\n🔍 进行RFM分析...")
        rfm_data = self.user_analyzer.calculate_rfm_features()
        rfm_segments = self.user_analyzer.rfm_segmentation()
        
        # 2. 高级特征计算
        print("\n🔍 计算高级用户特征...")
        user_features = self.user_analyzer.calculate_advanced_features()
        
        # 3. K-means聚类
        print("\n🔍 进行K-means聚类...")
        clusters = self.user_analyzer.kmeans_clustering()
        
        # 4. 生成用户画像
        print("\n🔍 生成用户画像...")
        profiles = self.user_analyzer.generate_user_profiles()
        
        # 5. 保存分析结果
        print("\n💾 保存分析结果...")
        self.user_analyzer.save_analysis_results('analysis_results')
        
        self.results['user_analysis'] = {
            'rfm_data': rfm_data,
            'rfm_segments': rfm_segments,
            'user_features': user_features,
            'user_profiles': profiles
        }
        
        print("✅ 用户分析完成!")
        return True
    
    def train_algorithms(self):
        """训练推荐算法"""
        if self.data is None:
            print("❌ 请先加载数据")
            return
            
        print("\n" + "="*50)
        print("🤖 开始训练推荐算法")
        print("="*50)
        
        # 1. 协同过滤算法
        print("\n🔍 训练协同过滤算法...")
        
        # 基于用户的协同过滤
        user_cf = CollaborativeFiltering('user_based')
        user_cf.prepare_data(self.data)
        user_cf.calculate_similarity()
        self.algorithms['user_cf'] = user_cf
        print("✅ 基于用户的协同过滤训练完成")
        
        # 基于物品的协同过滤
        item_cf = CollaborativeFiltering('item_based')
        item_cf.prepare_data(self.data)
        item_cf.calculate_similarity()
        self.algorithms['item_cf'] = item_cf
        print("✅ 基于物品的协同过滤训练完成")
        
        # 2. Transformer序列推荐
        print("\n🔍 训练Transformer序列推荐...")
        transformer_rec = TransformerRecommender(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=50
        )
        
        # 训练Transformer模型
        try:
            losses = transformer_rec.train(self.data, epochs=30, batch_size=128)
            self.algorithms['transformer'] = transformer_rec
            print("✅ Transformer序列推荐训练完成")
        except Exception as e:
            print(f"⚠️ Transformer训练失败: {e}")
        
        print(f"\n✅ 算法训练完成! 共训练 {len(self.algorithms)} 个算法")
        return True
    
    def evaluate_algorithms(self):
        """评估算法性能"""
        if not self.algorithms:
            print("❌ 请先训练算法")
            return
            
        print("\n" + "="*50)
        print("📊 开始算法性能评估")
        print("="*50)
        
        evaluation_results = {}
        
        # 选择测试用户
        test_users = self.data['user_id'].unique()[:100]  # 选择前100个用户进行测试
        
        for alg_name, algorithm in self.algorithms.items():
            print(f"\n📊 评估算法: {alg_name}")
            
            successful_recommendations = 0
            total_users = len(test_users)
            
            for user_id in test_users:
                try:
                    if alg_name in ['user_cf', 'item_cf']:
                        recommendations = algorithm.recommend(user_id, top_k=10)
                    elif alg_name == 'transformer':
                        recommendations = algorithm.recommend_for_user(self.data, user_id, top_k=10)
                    else:
                        recommendations = []
                    
                    if len(recommendations) > 0:
                        successful_recommendations += 1
                        
                except Exception as e:
                    continue
            
            coverage = successful_recommendations / total_users
            evaluation_results[alg_name] = {
                'coverage': coverage,
                'successful_recs': successful_recommendations,
                'total_users': total_users
            }
            
            print(f"   推荐覆盖率: {coverage:.2%}")
            print(f"   成功推荐用户数: {successful_recommendations}/{total_users}")
        
        self.results['evaluation'] = evaluation_results
        print("\n✅ 算法评估完成!")
        return evaluation_results
    
    def generate_recommendations(self, user_id, algorithm='auto', top_k=10):
        """为指定用户生成推荐"""
        if not self.algorithms:
            print("❌ 请先训练算法")
            return []
            
        print(f"\n🎯 为用户 {user_id} 生成推荐 (算法: {algorithm}, Top-{top_k})")
        
        recommendations = {}
        
        # 如果选择自动，则使用所有可用算法
        if algorithm == 'auto':
            algorithms_to_use = self.algorithms
        else:
            if algorithm in self.algorithms:
                algorithms_to_use = {algorithm: self.algorithms[algorithm]}
            else:
                print(f"❌ 算法 {algorithm} 不存在")
                return {}
        
        for alg_name, alg_instance in algorithms_to_use.items():
            try:
                if alg_name in ['user_cf', 'item_cf']:
                    recs = alg_instance.recommend(user_id, top_k=top_k)
                elif alg_name == 'transformer':
                    recs = alg_instance.recommend_for_user(self.data, user_id, top_k=top_k)
                else:
                    recs = []
                
                recommendations[alg_name] = recs
                print(f"✅ {alg_name}: {len(recs)} 个推荐")
                
            except Exception as e:
                print(f"❌ {alg_name} 推荐失败: {e}")
                recommendations[alg_name] = []
        
        return recommendations
    
    def get_user_profile(self, user_id):
        """获取用户画像"""
        if self.data is None:
            print("❌ 请先加载数据")
            return None
            
        user_data = self.data[self.data['user_id'] == user_id]
        
        if len(user_data) == 0:
            print(f"❌ 用户 {user_id} 不存在")
            return None
        
        # 计算用户基础特征
        profile = {
            'user_id': user_id,
            'total_actions': len(user_data),
            'unique_items': user_data['item_id'].nunique(),
            'behavior_distribution': dict(user_data['behavior_type'].value_counts()),
            'active_days': user_data['date'].nunique() if 'date' in user_data.columns else None,
            'first_action': user_data['datetime'].min() if 'datetime' in user_data.columns else None,
            'last_action': user_data['datetime'].max() if 'datetime' in user_data.columns else None
        }
        
        # 计算转化率
        behavior_counts = user_data['behavior_type'].value_counts()
        pv_count = behavior_counts.get('pv', 0)
        buy_count = behavior_counts.get('buy', 0)
        
        if pv_count > 0:
            profile['conversion_rate'] = buy_count / pv_count
        else:
            profile['conversion_rate'] = 0
        
        return profile
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("🚀 开始完整推荐系统分析流程")
        print("=" * 60)
        
        # 1. 数据加载
        if self.data is None:
            print("❌ 请先加载数据")
            return False
        
        # 2. 用户分析
        success = self.analyze_users()
        if not success:
            return False
        
        # 3. 算法训练
        success = self.train_algorithms()
        if not success:
            return False
        
        # 4. 算法评估
        self.evaluate_algorithms()
        
        # 5. 生成报告
        self.generate_summary_report()
        
        print(f"\n🎉 完整分析流程执行完成!")
        print(f"📁 结果已保存到 analysis_results 目录")
        
        return True
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n📋 生成分析总结报告...")
        
        report = []
        report.append("# 电商用户行为推荐系统分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## 数据概况")
        
        if self.data is not None:
            report.append(f"- 总记录数: {len(self.data):,}")
            report.append(f"- 用户数量: {self.data['user_id'].nunique():,}")
            report.append(f"- 商品数量: {self.data['item_id'].nunique():,}")
            
            behavior_dist = self.data['behavior_type'].value_counts()
            report.append(f"- 行为分布: {dict(behavior_dist)}")
        
        report.append("\n## 算法性能")
        if 'evaluation' in self.results:
            for alg_name, metrics in self.results['evaluation'].items():
                report.append(f"- {alg_name}: 覆盖率 {metrics['coverage']:.2%}")
        
        report.append("\n## 用户分群")
        if 'user_analysis' in self.results and 'user_profiles' in self.results['user_analysis']:
            profiles = self.results['user_analysis']['user_profiles']
            for segment, data in profiles.items():
                report.append(f"- {segment}: {data['用户数量']} 用户 ({data['占比']:.1f}%)")
        
        report.append("\n## 建议")
        report.append("1. 针对不同用户分群制定差异化推荐策略")
        report.append("2. 重点关注流失风险用户的挽回")
        report.append("3. 持续优化算法参数提升推荐效果")
        report.append("4. 结合业务场景选择合适的推荐算法")
        
        # 保存报告
        os.makedirs('analysis_results', exist_ok=True)
        with open('analysis_results/summary_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ 总结报告已保存到 analysis_results/summary_report.md")

def demo_quick_start():
    """快速开始演示"""
    print("🚀 推荐系统快速演示")
    print("=" * 50)
    
    # 创建示例数据
    print("📊 生成示例数据...")
    np.random.seed(42)
    
    n_users, n_items = 1000, 500
    data_list = []
    
    for _ in range(10000):
        user_id = np.random.randint(1, n_users + 1)
        item_id = np.random.randint(1, n_items + 1)
        behavior_type = np.random.choice(['pv', 'cart', 'fav', 'buy'], 
                                       p=[0.7, 0.15, 0.1, 0.05])
        timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        
        data_list.append({
            'user_id': user_id,
            'item_id': item_id,
            'behavior_type': behavior_type,
            'datetime': timestamp
        })
    
    demo_data = pd.DataFrame(data_list)
    demo_data.to_csv('demo_data.csv', index=False)
    print(f"✅ 示例数据已生成: demo_data.csv ({len(demo_data)} 条记录)")
    
    # 初始化推荐系统
    recommender = ECommerceRecommenderSystem('demo_data.csv')
    
    # 加载数据
    recommender.load_data()
    
    # 运行完整分析（简化版）
    print("\n🔍 运行用户分析...")
    recommender.analyze_users()
    
    print("\n🤖 训练推荐算法...")
    recommender.train_algorithms()
    
    print("\n📊 评估算法性能...")
    recommender.evaluate_algorithms()
    
    # 为示例用户生成推荐
    test_user = demo_data['user_id'].iloc[0]
    print(f"\n🎯 为用户 {test_user} 生成推荐...")
    recommendations = recommender.generate_recommendations(test_user, algorithm='auto', top_k=5)
    
    for alg_name, recs in recommendations.items():
        print(f"  {alg_name}: {recs}")
    
    # 生成报告
    recommender.generate_summary_report()
    
    print("\n✅ 快速演示完成!")
    print("💡 提示: 运行 'streamlit run recommendation_dashboard.py' 启动可视化界面")

def main():
    """主函数"""
    print("🛒 电商用户行为推荐系统")
    print("=" * 60)
    print("选择运行模式:")
    print("1. 快速演示 (使用示例数据)")
    print("2. 自定义数据分析")
    print("3. 启动可视化界面")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        demo_quick_start()
    elif choice == '2':
        data_path = input("请输入数据文件路径: ").strip()
        if os.path.exists(data_path):
            recommender = ECommerceRecommenderSystem(data_path)
            recommender.load_data()
            recommender.run_complete_analysis()
        else:
            print("❌ 文件不存在")
    elif choice == '3':
        print("💡 请运行以下命令启动可视化界面:")
        print("streamlit run recommendation_dashboard.py")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main() 