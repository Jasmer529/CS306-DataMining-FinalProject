"""
基于Transformer的序列推荐模型
实现SASRec (Self-Attentive Sequential Recommendation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SASRecModel(nn.Module):
    """SASRec Transformer推荐模型"""
    
    def __init__(self, n_items, hidden_size=64, num_layers=2, num_heads=2, 
                 max_seq_len=50, dropout=0.2):
        super().__init__()
        
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # 物品嵌入层
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, n_items)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, input_seq, mask=None):
        """前向传播"""
        # 获取序列长度
        seq_len = input_seq.size(1)
        
        # 物品嵌入
        embeddings = self.item_embedding(input_seq)  # [batch_size, seq_len, hidden_size]
        
        # 添加位置编码
        embeddings = self.pos_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # 创建因果注意力掩码（下三角矩阵）
        if mask is None:
            mask = self._generate_square_subsequent_mask(seq_len).to(input_seq.device)
        
        # Transformer编码
        hidden_states = self.transformer_encoder(embeddings, mask=mask)
        
        # 输出层
        logits = self.output_layer(hidden_states)  # [batch_size, seq_len, n_items]
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

class SequenceDataset(Dataset):
    """序列推荐数据集"""
    
    def __init__(self, sequences, max_seq_len=50):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 截取或填充序列到固定长度
        if len(sequence) > self.max_seq_len:
            sequence = sequence[-self.max_seq_len:]
        
        # 创建输入和目标序列
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        # 填充到最大长度
        input_seq = input_seq + [0] * (self.max_seq_len - 1 - len(input_seq))
        target_seq = target_seq + [0] * (self.max_seq_len - 1 - len(target_seq))
        
        return torch.tensor(input_seq), torch.tensor(target_seq)

class TransformerRecommender:
    """Transformer序列推荐系统"""
    
    def __init__(self, hidden_size=64, num_layers=2, num_heads=2, 
                 max_seq_len=50, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        
        self.model = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_sequences(self, data, min_seq_len=3):
        """准备用户序列数据"""
        print("🔍 准备用户行为序列...")
        
        # 按时间排序
        data_sorted = data.sort_values(['user_id', 'datetime'])
        
        # 创建物品索引映射
        unique_items = data_sorted['item_id'].unique()
        self.item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 0保留给padding
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # 构建用户序列
        user_sequences = []
        for user_id, user_data in data_sorted.groupby('user_id'):
            # 将物品ID转换为索引
            item_sequence = [self.item_to_idx[item] for item in user_data['item_id']]
            
            # 过滤短序列
            if len(item_sequence) >= min_seq_len:
                user_sequences.append(item_sequence)
        
        print(f"✅ 序列准备完成: {len(user_sequences)} 个用户序列")
        print(f"   物品数量: {len(unique_items)}")
        print(f"   平均序列长度: {np.mean([len(seq) for seq in user_sequences]):.1f}")
        
        return user_sequences
    
    def train(self, data, epochs=50, batch_size=256, min_seq_len=3):
        """训练Transformer模型"""
        print("🚀 开始训练Transformer推荐模型...")
        
        # 准备序列数据
        user_sequences = self.prepare_sequences(data, min_seq_len)
        
        # 创建数据集和数据加载器
        dataset = SequenceDataset(user_sequences, self.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        n_items = len(self.item_to_idx)
        self.model = SASRecModel(
            n_items=n_items,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_seq_len=self.max_seq_len
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # 训练循环
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits = self.model(input_seq)
                
                # 计算损失
                loss = criterion(logits.view(-1, logits.size(-1)), target_seq.view(-1))
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        print(f"✅ 训练完成，最终损失: {losses[-1]:.4f}")
        return losses
    
    def predict_next_items(self, user_sequence, top_k=10):
        """预测用户序列的下一个物品"""
        if self.model is None:
            print("❌ 请先训练模型")
            return []
        
        self.model.eval()
        
        # 转换物品ID为索引
        try:
            sequence_indices = [self.item_to_idx.get(item, 0) for item in user_sequence]
        except:
            return []
        
        # 截取或填充序列
        if len(sequence_indices) > self.max_seq_len - 1:
            sequence_indices = sequence_indices[-(self.max_seq_len - 1):]
        
        # 填充到固定长度
        padded_sequence = sequence_indices + [0] * (self.max_seq_len - 1 - len(sequence_indices))
        input_tensor = torch.tensor([padded_sequence]).to(self.device)
        
        with torch.no_grad():
            # 获取最后一个时间步的预测
            logits = self.model(input_tensor)
            last_logits = logits[0, len(sequence_indices) - 1, :]  # 最后一个有效位置
            
            # 获取top-k预测
            _, top_indices = torch.topk(last_logits, top_k)
            
            # 转换回物品ID
            recommendations = []
            for idx in top_indices.cpu().numpy():
                if idx > 0 and idx in self.idx_to_item:  # 排除padding索引
                    item_id = self.idx_to_item[idx]
                    if item_id not in user_sequence:  # 避免推荐已交互物品
                        recommendations.append(item_id)
            
            return recommendations[:top_k]
    
    def recommend_for_user(self, data, user_id, top_k=10):
        """为指定用户推荐物品"""
        # 获取用户历史序列
        user_data = data[data['user_id'] == user_id].sort_values('datetime')
        
        if len(user_data) == 0:
            return []
        
        user_sequence = user_data['item_id'].tolist()
        return self.predict_next_items(user_sequence, top_k)
    
    def evaluate_model(self, data, test_ratio=0.2):
        """评估模型性能"""
        print("📊 评估模型性能...")
        
        # 准备测试数据
        user_sequences = self.prepare_sequences(data)
        
        hit_count = 0
        total_count = 0
        
        for sequence in user_sequences:
            if len(sequence) < 4:  # 需要足够长的序列进行测试
                continue
            
            # 使用前80%作为输入，后20%作为测试目标
            split_point = int(len(sequence) * (1 - test_ratio))
            input_seq = [self.idx_to_item.get(idx, 0) for idx in sequence[:split_point]]
            target_items = set([self.idx_to_item.get(idx, 0) for idx in sequence[split_point:]])
            
            # 获取推荐结果
            recommendations = self.predict_next_items(input_seq, top_k=10)
            
            # 计算命中率
            if any(item in target_items for item in recommendations):
                hit_count += 1
            total_count += 1
        
        hit_rate = hit_count / total_count if total_count > 0 else 0
        print(f"✅ 模型评估完成:")
        print(f"   测试序列数: {total_count}")
        print(f"   命中率 (Hit Rate): {hit_rate:.3f}")
        
        return hit_rate
    
    def visualize_attention_weights(self, user_sequence, save_path=None):
        """可视化注意力权重"""
        if self.model is None:
            print("❌ 请先训练模型")
            return
        
        # 这里简化处理，实际需要修改模型以返回注意力权重
        print("💡 注意力权重可视化功能需要模型修改以暴露注意力层")
        print("   建议在模型forward方法中返回注意力权重")
    
    def save_model(self, path):
        """保存模型"""
        if self.model is None:
            print("❌ 没有可保存的模型")
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'model_params': {
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'max_seq_len': self.max_seq_len
            }
        }
        
        torch.save(checkpoint, path)
        print(f"✅ 模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 恢复映射关系
        self.item_to_idx = checkpoint['item_to_idx']
        self.idx_to_item = checkpoint['idx_to_item']
        
        # 重建模型
        params = checkpoint['model_params']
        n_items = len(self.item_to_idx)
        
        self.model = SASRecModel(
            n_items=n_items,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            max_seq_len=params['max_seq_len']
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型已从 {path} 加载")

def demo_transformer_recommender():
    """Transformer推荐演示"""
    print("🚀 Transformer序列推荐演示")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    n_users, n_items = 1000, 500
    
    data_list = []
    for user_id in range(1, n_users + 1):
        # 为每个用户生成时间序列行为
        n_actions = np.random.randint(5, 50)
        items = np.random.choice(range(1, n_items + 1), size=n_actions, replace=True)
        
        for i, item_id in enumerate(items):
            timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
            data_list.append({
                'user_id': user_id,
                'item_id': item_id,
                'datetime': timestamp,
                'behavior_type': 'pv'
            })
    
    demo_data = pd.DataFrame(data_list)
    print(f"📊 示例数据: {len(demo_data)} 条行为记录")
    
    # 初始化Transformer推荐器
    recommender = TransformerRecommender(
        hidden_size=32,  # 为了演示使用较小的模型
        num_layers=2,
        num_heads=2,
        max_seq_len=20
    )
    
    # 训练模型
    print("\n🔍 训练Transformer模型...")
    losses = recommender.train(demo_data, epochs=20, batch_size=64)
    
    # 为用户推荐
    test_user = 1
    recommendations = recommender.recommend_for_user(demo_data, test_user, top_k=5)
    print(f"\n📋 为用户 {test_user} 推荐的物品: {recommendations}")
    
    # 评估模型
    hit_rate = recommender.evaluate_model(demo_data)
    
    # 可视化损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Transformer模型训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    print("\n✅ Transformer推荐演示完成!")

if __name__ == "__main__":
    demo_transformer_recommender() 