# 🛒 电商用户行为推荐系统

基于深度学习和用户行为分析的个性化推荐系统，参考淘宝用户行为数据集实现。

## 📋 项目概述

本项目实现了一个完整的电商推荐系统，包含用户行为分析、多种推荐算法、用户分群、可视化界面等功能。适用于电商平台的个性化推荐、用户画像构建、商品推荐等场景。

## ✨ 核心功能

### 🔍 用户行为分析
- **RFM分析**: 基于最近性、频率、货币价值的用户分群
- **用户画像**: 多维度用户特征提取和标签体系
- **行为模式挖掘**: 转化率分析、活跃时段分析、偏好分析
- **K-means聚类**: 基于用户特征的无监督聚类

### 🤖 推荐算法
- **协同过滤**: 基于用户和基于物品的协同过滤
- **矩阵分解**: SVD、NMF等矩阵分解技术
- **序列推荐**: LSTM、Transformer序列建模
- **深度学习**: 神经协同过滤、Wide&Deep等
- **混合推荐**: 多算法融合策略

### 📊 可视化分析
- **交互式界面**: Streamlit构建的Web界面
- **数据概览**: 用户、商品、行为分布分析
- **性能对比**: 算法性能雷达图、指标对比
- **个性化推荐**: 实时推荐生成和解释



## 🎯 启动方式：交互式仪表板

运行可视化仪表板，最适合演示和分析：

```bash
streamlit run recommendation_dashboard.py
```

启动后会自动打开浏览器。

**功能包括：**

- 数据概览和可视化
- 用户行为分析
- 用户分群
- 算法性能对比
- 个性化推荐演示

**📊 数据文件支持：**

- **支持最大 5GB 的 CSV 文件上传**
- 原始 UserBehavior.csv (3.4GB) ✅
- 预处理后的数据文件 ✅
- 自定义数据文件 ✅
- 大文件加载会显示进度条

## 📁 项目结构

```
📦 推荐系统项目
├── 📄 main_recommender_system.py      # 主程序入口
├── 📄 user_analysis_enhancement.py    # 用户分析模块
├── 📄 collaborative_filtering.py      # 协同过滤算法
├── 📄 transformer_recommender.py      # Transformer序列推荐
├── 📄 recommendation_dashboard.py     # 可视化界面
├── 📄 development_roadmap.md          # 开发路线图
├── 📄 requirements.txt               # 依赖包列表
├── 📄 README.md                      # 项目说明
└── 📂 analysis_results/              # 分析结果输出
    ├── 📄 user_features.csv
    ├── 📄 rfm_analysis.csv
    ├── 📄 user_profiles.csv
    └── 📄 summary_report.md
```

