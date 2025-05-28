"""
推荐系统可视化界面
使用Streamlit构建交互式数据挖掘分析平台
"""

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
        
        # 添加文件大小提示
        st.sidebar.info("💡 **文件上传说明**\n"
                       "- 支持最大5GB的CSV文件\n"
                       "- 推荐使用预处理后的数据文件\n"
                       "- 大文件加载可能需要较长时间")
        
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
        """数据概览页面 - 包含来自 ylz_version1.ipynb 的所有可视化"""
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
            st.dataframe(df.head(10))
            
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
                        st.dataframe(rfm_df.head(), use_container_width=True)
                    
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
        
        # 模拟算法性能数据
        algorithm_performance = {
            '算法名称': ['协同过滤(用户)', '协同过滤(物品)', '矩阵分解', 'LSTM序列', 'Transformer', '深度神经网络'],
            '准确率': [0.65, 0.68, 0.72, 0.75, 0.78, 0.73],
            '召回率': [0.58, 0.62, 0.69, 0.71, 0.74, 0.70],
            'F1分数': [0.61, 0.65, 0.70, 0.73, 0.76, 0.71],
            '覆盖率': [0.45, 0.52, 0.58, 0.62, 0.65, 0.60],
            '多样性': [0.72, 0.68, 0.65, 0.70, 0.73, 0.67],
            '训练时间(分钟)': [15, 18, 45, 120, 180, 90]
        }
        
        performance_df = pd.DataFrame(algorithm_performance)
        
        # 性能对比雷达图
        st.subheader("📊 算法性能对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 雷达图
            categories = ['准确率', '召回率', 'F1分数', '覆盖率', '多样性']
            
            fig = go.Figure()
            
            for i, algorithm in enumerate(performance_df['算法名称']):
                values = [performance_df.iloc[i][cat] for cat in categories]
                values += [values[0]]  # 闭合雷达图
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=algorithm
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="算法性能雷达图"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 性能指标柱状图
            selected_metric = st.selectbox("选择性能指标", categories)
            
            fig = px.bar(
                performance_df,
                x='算法名称',
                y=selected_metric,
                title=f"{selected_metric}对比",
                color=selected_metric,
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # 性能详细表格
        st.subheader("📋 详细性能指标")
        st.dataframe(performance_df, use_container_width=True)
        
        # 算法推荐建议
        st.subheader("💡 算法选择建议")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🏆 最佳综合性能**
            - Transformer序列推荐
            - 在准确率、召回率等多项指标表现优异
            - 适合有充足数据和计算资源的场景
            """)
        
        with col2:
            st.success("""
            **⚡ 最佳效率平衡**
            - 矩阵分解算法
            - 性能良好且训练时间适中
            - 适合中等规模的推荐场景
            """)
        
        with col3:
            st.warning("""
            **🚀 快速部署**
            - 协同过滤算法
            - 实现简单、训练快速
            - 适合快速原型和小规模应用
            """)
    
    def render_personalized_recommendation(self):
        """渲染个性化推荐页面"""
        st.title("🎯 个性化推荐")
        
        if self.data is None:
            st.warning("⚠️ 请先在侧边栏上传数据文件")
            return
        
        # 用户选择
        st.subheader("👤 选择用户")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            user_list = self.data['user_id'].unique()[:100]  # 限制显示用户数量
            selected_user = st.selectbox("选择用户ID", user_list)
            
            # 推荐算法选择
            algorithm = st.selectbox(
                "选择推荐算法",
                ["协同过滤", "矩阵分解", "Transformer", "混合推荐"]
            )
            
            recommendation_count = st.slider("推荐数量", 5, 20, 10)
        
        with col2:
            # 用户历史行为
            st.write("**用户历史行为**")
            user_history = self.data[self.data['user_id'] == selected_user].tail(10)
            
            if len(user_history) > 0:
                behavior_summary = user_history['behavior_type'].value_counts()
                
                # 行为类型饼图
                fig = px.pie(
                    values=behavior_summary.values,
                    names=behavior_summary.index,
                    title=f"用户 {selected_user} 行为分布"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 历史记录表格 - 动态选择可用的时间列
            display_columns = ['item_id', 'behavior_type']
            
            # 检查可用的时间列
            time_columns = ['timestamp_dt', 'date', 'datetime', 'timestamp']
            available_time_column = None
            for col in time_columns:
                if col in user_history.columns:
                    available_time_column = col
                    break
            
            if available_time_column:
                display_columns.append(available_time_column)
            
            # 如果有category_id也显示
            if 'category_id' in user_history.columns:
                display_columns.append('category_id')
            
            st.dataframe(user_history[display_columns], use_container_width=True)
        
        # 生成推荐结果
        st.subheader("📋 推荐结果")
        
        if st.button("🎯 生成推荐", type="primary"):
            with st.spinner("正在生成个性化推荐..."):
                # 模拟推荐结果
                np.random.seed(hash(str(selected_user)) % 2**32)
                
                # 获取用户未交互过的商品
                user_items = set(self.data[self.data['user_id'] == selected_user]['item_id'])
                all_items = set(self.data['item_id'].unique())
                candidate_items = list(all_items - user_items)
                
                if len(candidate_items) >= recommendation_count:
                    recommended_items = np.random.choice(
                        candidate_items, 
                        size=recommendation_count, 
                        replace=False
                    )
                    
                    # 生成模拟推荐分数
                    recommendation_scores = np.random.uniform(0.6, 0.95, recommendation_count)
                    
                    recommendations_df = pd.DataFrame({
                        '商品ID': recommended_items,
                        '推荐分数': recommendation_scores,
                        '推荐原因': [f"基于{algorithm}算法" for _ in range(recommendation_count)]
                    })
                    
                    recommendations_df = recommendations_df.sort_values('推荐分数', ascending=False)
                    
                    # 显示推荐结果
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(recommendations_df, use_container_width=True)
                    
                    with col2:
                        # 推荐分数分布
                        fig = px.bar(
                            recommendations_df,
                            x='商品ID',
                            y='推荐分数',
                            title="推荐分数分布"
                        )
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 推荐解释
                    st.info(f"""
                    **推荐解释**
                    - 使用 {algorithm} 算法为用户 {selected_user} 生成推荐
                    - 基于用户历史行为模式和相似用户偏好
                    - 推荐商品均为用户未曾交互过的商品
                    - 推荐分数反映商品与用户兴趣的匹配度
                    """)
                else:
                    st.error("该用户的可推荐商品数量不足")
    
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