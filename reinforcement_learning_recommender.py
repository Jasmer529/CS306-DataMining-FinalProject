"""
基于强化学习的推荐系统
使用深度Q网络(DQN)学习推荐策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
import warnings
warnings.filterwarnings('ignore')

# 定义经验回放的转换结构
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """保存转换"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """随机采样"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class RecommendationEnvironment:
    """推荐环境"""
    
    def __init__(self, data, state_dim=50, max_session_length=20):
        self.data = data
        self.state_dim = state_dim
        self.max_session_length = max_session_length
        
        # 创建用户和商品的映射
        self.users = sorted(data['user_id'].unique())
        self.items = sorted(data['item_id'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(self.users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.items)}
        
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        
        # 准备用户历史数据
        self.user_histories = self._prepare_user_histories()
        
        # 计算商品特征
        self.item_features = self._compute_item_features()
        
        # 当前状态
        self.current_user = None
        self.current_session = []
        self.session_step = 0
        
        print(f"🎮 推荐环境初始化完成:")
        print(f"   用户数量: {self.num_users}")
        print(f"   商品数量: {self.num_items}")
        print(f"   状态维度: {self.state_dim}")
    
    def _prepare_user_histories(self):
        """准备用户历史行为数据"""
        user_histories = {}
        
        for user_id in self.users:
            user_data = self.data[self.data['user_id'] == user_id].sort_values('datetime')
            
            # 提取用户行为序列
            history = []
            for _, row in user_data.iterrows():
                item_idx = self.item_to_idx[row['item_id']]
                behavior_type = row['behavior_type']
                
                # 将行为类型转换为数值
                behavior_score = {
                    'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4
                }.get(behavior_type, 1)
                
                history.append((item_idx, behavior_score))
            
            user_histories[user_id] = history
        
        return user_histories
    
    def _compute_item_features(self):
        """计算商品特征向量"""
        item_features = {}
        
        # 基本统计特征
        item_stats = self.data.groupby('item_id').agg({
            'user_id': 'nunique',  # 用户数
            'behavior_type': ['count', lambda x: (x == 'buy').sum()]  # 总行为数，购买数
        }).round(4)
        
        item_stats.columns = ['unique_users', 'total_behaviors', 'total_purchases']
        item_stats['purchase_rate'] = item_stats['total_purchases'] / item_stats['total_behaviors']
        item_stats['popularity'] = item_stats['total_behaviors'] / item_stats['total_behaviors'].max()
        
        # 品类特征
        if 'category_id' in self.data.columns:
            category_mapping = self.data.groupby('item_id')['category_id'].first()
            item_stats['category'] = category_mapping
        else:
            item_stats['category'] = 0
        
        # 归一化特征
        for col in ['unique_users', 'total_behaviors', 'total_purchases']:
            max_val = item_stats[col].max()
            if max_val > 0:
                item_stats[col] = item_stats[col] / max_val
        
        # 转换为特征向量
        for item_id in self.items:
            if item_id in item_stats.index:
                features = [
                    item_stats.loc[item_id, 'unique_users'],
                    item_stats.loc[item_id, 'total_behaviors'],
                    item_stats.loc[item_id, 'total_purchases'],
                    item_stats.loc[item_id, 'purchase_rate'],
                    item_stats.loc[item_id, 'popularity']
                ]
            else:
                features = [0.0] * 5
            
            # 扩展到固定维度
            while len(features) < 10:
                features.append(0.0)
            
            item_features[self.item_to_idx[item_id]] = features[:10]
        
        return item_features
    
    def reset(self, user_id=None):
        """重置环境"""
        if user_id is None:
            self.current_user = random.choice(self.users)
        else:
            self.current_user = user_id
        
        self.current_session = []
        self.session_step = 0
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态向量"""
        state = np.zeros(self.state_dim)
        
        # 用户历史特征
        user_history = self.user_histories.get(self.current_user, [])
        
        # 最近交互的商品特征 (前20个)
        recent_items = user_history[-20:] if len(user_history) >= 20 else user_history
        
        idx = 0
        for item_idx, behavior_score in recent_items:
            if idx + 10 < self.state_dim:
                # 添加商品特征
                if item_idx in self.item_features:
                    features = self.item_features[item_idx]
                    state[idx:idx+10] = features
                
                # 添加行为评分
                if idx + 10 < self.state_dim:
                    state[idx + 10] = behavior_score / 4.0  # 归一化
                
                idx += 11
            else:
                break
        
        # 当前会话特征
        if self.current_session:
            session_length = len(self.current_session)
            if self.state_dim > 45:
                state[-5] = min(session_length / self.max_session_length, 1.0)
                
                # 最近推荐的商品
                last_items = self.current_session[-4:]
                for i, item_idx in enumerate(last_items):
                    if i < 4:
                        state[-4 + i] = item_idx / self.num_items
        
        return state
    
    def step(self, action):
        """执行动作"""
        # action 是推荐的商品索引
        recommended_item_idx = action
        recommended_item_id = self.items[recommended_item_idx]
        
        # 计算奖励
        reward = self._calculate_reward(recommended_item_id)
        
        # 更新会话
        self.current_session.append(recommended_item_idx)
        self.session_step += 1
        
        # 判断是否结束
        done = (self.session_step >= self.max_session_length)
        
        next_state = self._get_state()
        
        return next_state, reward, done
    
    def _calculate_reward(self, recommended_item_id):
        """计算推荐奖励"""
        user_history = self.user_histories.get(self.current_user, [])
        
        # 基础奖励：商品受欢迎程度
        item_idx = self.item_to_idx[recommended_item_id]
        base_reward = self.item_features.get(item_idx, [0] * 10)[4]  # 流行度
        
        # 个性化奖励：用户是否交互过该商品
        user_items = [item for item, _ in user_history]
        if item_idx in user_items:
            # 找到用户对该商品的最高评分
            max_score = max([score for item, score in user_history if item == item_idx])
            personalized_reward = max_score / 4.0  # 归一化到[0,1]
        else:
            # 新商品的探索奖励
            personalized_reward = 0.1
        
        # 多样性奖励：避免重复推荐
        diversity_reward = 0.0
        if item_idx not in self.current_session:
            diversity_reward = 0.2
        
        # 总奖励
        total_reward = base_reward * 0.4 + personalized_reward * 0.5 + diversity_reward * 0.1
        
        return total_reward

class RLRecommenderAgent:
    """强化学习推荐智能体"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Q网络
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = ReplayBuffer(capacity=10000)
        
        # 训练参数
        self.batch_size = 64
        self.gamma = 0.95  # 折扣因子
        self.target_update_freq = 100  # 目标网络更新频率
        self.step_count = 0
        
        print(f"🤖 强化学习智能体初始化完成 (设备: {self.device})")
    
    def act(self, state, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            # ε-贪婪探索
            return random.randrange(self.action_dim)
        
        # 利用策略
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action
    
    def remember(self, state, action, next_state, reward):
        """存储经验"""
        self.memory.push(state, action, next_state, reward)
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 采样批次
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # 转换为张量
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新ε值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class RLRecommenderSystem:
    """强化学习推荐系统"""
    
    def __init__(self, state_dim=50, hidden_dim=128, learning_rate=0.001):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        self.env = None
        self.agent = None
        self.training_history = []
        
        print("🎯 强化学习推荐系统初始化")
    
    def setup(self, data):
        """设置环境和智能体"""
        print("🔧 设置推荐环境和智能体...")
        
        # 创建环境
        self.env = RecommendationEnvironment(data, state_dim=self.state_dim)
        
        # 创建智能体
        self.agent = RLRecommenderAgent(
            state_dim=self.state_dim,
            action_dim=self.env.num_items,
            learning_rate=self.learning_rate
        )
        
        print("✅ 环境和智能体设置完成")
    
    def train(self, episodes=1000, max_steps_per_episode=20):
        """训练强化学习智能体"""
        if self.env is None or self.agent is None:
            print("❌ 请先设置环境和智能体")
            return
        
        print(f"\n🚀 开始强化学习训练 (episodes={episodes})")
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(episodes):
            # 随机选择用户开始新的回合
            state = self.env.reset()
            total_reward = 0.0
            total_loss = 0.0
            step_count = 0
            
            for step in range(max_steps_per_episode):
                # 选择动作
                action = self.agent.act(state, training=True)
                
                # 执行动作
                next_state, reward, done = self.env.step(action)
                
                # 存储经验
                self.agent.remember(state, action, next_state, reward)
                
                # 训练智能体
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    total_loss += loss
                    step_count += 1
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
                
                # 更新目标网络
                self.agent.step_count += 1
                if self.agent.step_count % self.agent.target_update_freq == 0:
                    self.agent.update_target_network()
            
            avg_loss = total_loss / max(step_count, 1)
            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)
            
            # 记录训练历史
            self.training_history.append({
                'episode': episode,
                'reward': total_reward,
                'loss': avg_loss,
                'epsilon': self.agent.epsilon
            })
            
            # 打印进度
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(episode_losses[-100:])
                print(f"   Episode {episode+1}/{episodes} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}, ε: {self.agent.epsilon:.4f}")
        
        print("✅ 强化学习训练完成!")
        return episode_rewards, episode_losses
    
    def recommend_for_user(self, user_id, top_k=10):
        """为用户生成推荐"""
        if self.env is None or self.agent is None:
            print("❌ 请先设置并训练系统")
            return []
        
        if user_id not in self.env.user_to_idx:
            print(f"❌ 用户 {user_id} 不在训练数据中")
            return []
        
        # 重置环境到指定用户
        state = self.env.reset(user_id)
        
        recommendations = []
        used_actions = set()
        
        # 生成推荐序列
        for _ in range(top_k):
            # 获取Q值
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            
            with torch.no_grad():
                q_values = self.agent.q_network(state_tensor).squeeze()
            
            # 选择未推荐过的最优动作
            q_values_np = q_values.cpu().numpy()
            
            # 将已使用的动作的Q值设为负无穷
            for used_action in used_actions:
                q_values_np[used_action] = float('-inf')
            
            if np.all(np.isinf(q_values_np)):
                break
            
            action = np.argmax(q_values_np)
            item_id = self.env.items[action]
            
            recommendations.append(item_id)
            used_actions.add(action)
            
            # 更新状态
            state, _, _ = self.env.step(action)
        
        return recommendations
    
    def evaluate_recommendations(self, test_users, top_k=10):
        """评估推荐性能"""
        if not test_users:
            return {}
        
        print(f"\n📊 评估强化学习推荐性能 (用户数: {len(test_users)})")
        
        total_diversity = 0.0
        total_coverage = set()
        successful_recommendations = 0
        
        for user_id in test_users:
            try:
                recommendations = self.recommend_for_user(user_id, top_k)
                
                if recommendations:
                    # 多样性：推荐列表中的唯一商品数
                    diversity = len(set(recommendations)) / len(recommendations)
                    total_diversity += diversity
                    
                    # 覆盖率：推荐的唯一商品
                    total_coverage.update(recommendations)
                    
                    successful_recommendations += 1
                    
            except Exception as e:
                continue
        
        # 计算指标
        results = {}
        if successful_recommendations > 0:
            results['avg_diversity'] = total_diversity / successful_recommendations
            results['coverage'] = len(total_coverage) / self.env.num_items
            results['success_rate'] = successful_recommendations / len(test_users)
        else:
            results['avg_diversity'] = 0.0
            results['coverage'] = 0.0
            results['success_rate'] = 0.0
        
        print(f"   📈 平均多样性: {results['avg_diversity']:.4f}")
        print(f"   📊 覆盖率: {results['coverage']:.4f}")
        print(f"   ✅ 成功率: {results['success_rate']:.4f}")
        
        return results
    
    def save_model(self, path):
        """保存模型"""
        if self.agent is None:
            print("❌ 没有训练好的模型可以保存")
            return
        
        torch.save({
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.agent.action_dim,
                'learning_rate': self.learning_rate
            }
        }, path)
        
        print(f"💾 强化学习模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.agent.device if self.agent else 'cpu')
        
        if self.agent is None:
            print("❌ 请先设置环境和智能体")
            return
        
        self.agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"📂 强化学习模型已从 {path} 加载") 