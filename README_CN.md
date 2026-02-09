# DiaMetric-CDC: 基于CDC BRFSS数据的糖尿病风险智能预测系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-orange.svg)

**一个端到端的机器学习项目，用于糖尿病风险分层和预测建模**

[English](README.md) | 简体中文

</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [技术架构](#-技术架构)
- [数据集](#-数据集)
- [项目结构](#-项目结构)
- [环境配置](#-环境配置)
- [快速开始](#-快速开始)
- [五阶段分析流程](#-五阶段分析流程)
- [核心成果](#-核心成果)
- [可解释性分析](#-可解释性分析)
- [公共卫生应用](#-公共卫生应用)
- [技术亮点](#-技术亮点)
- [未来展望](#-未来展望)

---

## 🎯 项目概述

**DiaMetric-CDC** 是一个基于2015年CDC BRFSS（行为风险因素监测系统）数据的糖尿病风险预测系统。该项目采用**无监督聚类**和**监督分类**相结合的混合建模策略，实现从数据理解、特征工程、人群分层到风险预测的完整机器学习流程。

### 核心目标

- **智能风险分层**：通过K-Prototypes聚类算法识别6种潜在的糖尿病风险表型
- **高精度预测**：基于XGBoost构建校准的二分类模型（AUC-ROC: 0.819）
- **临床可解释性**：使用SHAP值提供特征级别的全局和局部解释
- **公共卫生决策支持**：为不同风险群体制定针对性干预策略

### 应用场景

- 🏥 **医疗机构**：患者风险筛查和临床决策支持
- 🏛️ **公共卫生部门**：人群健康监测和资源分配优化
- 📊 **健康数据科学**：大规模流行病学数据分析范例
- 🎓 **学术研究**：糖尿病社会决定因素和行为风险因素研究

---

## ✨ 核心特性

### 🔬 先进的分析方法

- **混合数据类型处理**：K-Prototypes算法支持连续和分类特征的统一聚类
- **类别不平衡处理**：成本敏感学习（FN:FP = 5:1）和样本加权策略
- **超参数自动优化**：基于OPTUNA框架的贝叶斯优化（50次试验）
- **概率校准**：等渗回归提升预测概率可靠性（ECE: 0.007）
- **可解释AI**：SHAP值驱动的特征重要性和决策路径分析

### 📊 全面的模型评估

- **消融实验**：量化聚类特征对模型性能的贡献（+0.56%临床评分提升）
- **阈值优化**：针对不同临床场景的多阈值策略（默认/约登指数/高特异性）
- **稳健性验证**：5折分层交叉验证和Bootstrap置信区间（95% CI）
- **公平性审计**：按性别和收入分层的模型性能评估

### 🎨 丰富的可视化

- **18个分类模型图表**：ROC/PR曲线、校准曲线、混淆矩阵、SHAP依赖图等
- **10个聚类分析图表**：UMAP/t-SNE降维、雷达图、热力图、决策树代理模型
- **交互式HTML报告**：完整的EDA、聚类和分类建模分析报告

---

## 🏗️ 技术架构

### 核心框架

```
数据层              → 处理层                → 特征层              → 模型层               → 应用层
├─ CDC BRFSS       → ├─ 去重聚合           → ├─ 临床离散化       → ├─ K-Prototypes    → ├─ 风险评分
│  (253,680条)     → ├─ 逻辑一致性检查     → ├─ 交互特征         → ├─ XGBoost         → ├─ 干预策略
├─ 21个原始特征    → ├─ 目标二值化         → ├─ 聚合指数         → │  (24基线+2聚类)  → ├─ SHAP解释
└─ Sample_Weight   → └─ 权重标准化         → └─ VIF筛选          → ├─ 概率校准         → └─ 公平性审计
                                                               → └─ 阈值优化
```

### 技术栈

| 类别 | 技术 |
|------|------|
| **核心语言** | Python 3.8+ |
| **数据处理** | Pandas, NumPy |
| **机器学习** | Scikit-learn, XGBoost, LightGBM, Imbalanced-learn |
| **聚类算法** | K-Prototypes, UMAP |
| **超参数优化** | Optuna |
| **可解释AI** | SHAP |
| **加速计算** | Scikit-learn-intelex (Intel Extension), CUDA (GPU) |
| **可视化** | Matplotlib, Seaborn |
| **统计分析** | SciPy, Statsmodels |

---

## 📊 数据集

### 数据源

**数据集名称**: 2015 CDC BRFSS Diabetes Health Indicators  
**数据来源**: Centers for Disease Control and Prevention (CDC)  
**记录数量**: 253,680条调查响应  
**特征数量**: 21个行为和健康指标 + 1个目标变量  
**类别分布**: 
- 非糖尿病: 213,703 (84.2%)
- 前驱糖尿病/糖尿病: 39,977 (15.8%)
- **类别不平衡比**: 4.79:1

### 特征类别

| 类别 | 特征 |
|------|------|
| **生理指标** (4) | BMI, HighBP, HighChol, CholCheck |
| **慢性疾病** (2) | HeartDiseaseorAttack, Stroke |
| **主观健康** (3) | GenHlth, MentHlth, PhysHlth |
| **生活方式** (4) | Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump |
| **医疗可及性** (2) | AnyHealthcare, NoDocbcCost |
| **功能状态** (1) | DiffWalk |
| **社会经济学** (3) | Income, Education, Age |
| **人口统计学** (1) | Sex |

### 数据处理流程

```
原始数据(253,680条) → 逻辑一致性清洗(253,264条) 
  → Profile去重聚合(229,296条唯一配置) 
  → 二值化目标(Diabetes_binary) → 特征工程(24基线特征) 
  → 聚类增强(+Risk_Index/Cluster_ID，共26特征)
```

**说明**: 
1. **逻辑清洗**: 移除416条矛盾记录（如`CholCheck=0 & HighChol=1`），253,680→253,264365
2. **Profile去重**: 基于21特征+目标的唯一组合，聚合229,365条为229,296条配置（合并69条具有相同特征但权重不同的记录）
3. **权重保持**: 每条记录的`Sample_Weight`保留原始频次，加权总和=253,264

---

## 📁 项目结构

```
DiaMetric-CDC/
│
├── 01_Data_Understanding_EDA.ipynb           # Phase 1: 探索性数据分析
├── 02_Data_Preprocessing.ipynb               # Phase 2: 数据预处理与质量控制
├── 03_Feature_Engineering.ipynb              # Phase 3: 特征工程与数据分割
├── 04_Clustering_K-Prototypes.ipynb          # Phase 4: K-Prototypes聚类分析
├── 05_Classification_Modeling.ipynb          # Phase 5: 分类建模与模型评估
├── requirements.txt                          # Python依赖包列表
│
├── data/                                     # 数据目录
│   ├── raw/                                  
│   │   └── CDC Diabetes Dataset.csv         # 原始CDC数据集
│   └── processed/                            
│       ├── data_preprocessing/               
│       │   └── CDC_Diabetes_Cleaned.csv     # 清洗后数据
│       ├── feature_engineering/              
│       │   ├── CDC_Train_Classification_BASELINE.csv  # 基线训练集
│       │   ├── CDC_Test_Classification_BASELINE.csv   # 基线测试集
│       │   ├── CDC_Clustering_RAW.csv       # 聚类原始特征
│       │   └── CDC_Clustering_SCALED.csv    # 聚类标准化特征
│       └── clustering_k-prototypes/          
│           ├── CDC_Train_Classification_CLUSTERED.csv  # 聚类增强训练集
│           └── CDC_Test_Classification_CLUSTERED.csv   # 聚类增强测试集
│
├── outputs/                                  # 输出结果目录
│   ├── classification/                       
│   │   ├── images/                           # 18张分类建模图表
│   │   ├── models/                           
│   │   │   ├── champion_model_calibrated.pkl # 校准后最优模型（推荐）
│   │   │   ├── champion_model_uncalibrated.pkl # 未校准模型
│   │   │   ├── champion_model_metadata.json # 模型元数据
│   │   │   ├── inference_pipeline.py        # 推理脚本
│   │   │   └── feature_configuration.json   # 特征配置
│   │   ├── tables/                           
│   │   │   ├── shap_feature_importance.csv  # SHAP特征重要性
│   │   │   ├── ablation_benchmark_results.csv  # 消融实验结果
│   │   │   ├── calibration_methods_comparison.csv  # 校准对比
│   │   │   └── Risk_Probabilities.csv       # 测试集预测概率
│   │   └── logs/                             # 训练日志
│   │
│   ├── clustering_k-prototypes/              
│   │   ├── images/                           # 10张聚类分析图表
│   │   ├── models/                           
│   │   │   └── optimal_gamma.json           # 最优Gamma参数
│   │   ├── tables/                           
│   │   │   ├── cluster_profiles.json        # 聚类特征画像
│   │   │   └── k_optimization_results.csv   # K值优化结果
│   │   └── targeted_intervention_policy_report.txt  # 干预策略报告
│   │
│   ├── feature_engineering/                  
│   │   └── feature_metadata.json            # 特征元数据
│   │
│   └── data_understanding/                   # EDA可视化结果
```

---

## 🔧 环境配置

### 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python版本**: 3.8 或更高
- **内存**: 建议 ≥16GB RAM
- **GPU（可选）**: NVIDIA GPU + CUDA 12.1（用于XGBoost加速）

### 依赖安装

#### 方法一：使用requirements.txt（推荐）

```bash
# 克隆仓库（替换为你的实际仓库地址）
# git clone https://github.com/yourusername/DiaMetric-CDC.git
# cd DiaMetric-CDC

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖（包含GPU支持的PyTorch）
pip install -r requirements.txt
```

#### 方法二：仅CPU环境

```bash
# 编辑requirements.txt，移除--index-url行和torch相关包
# 然后执行：
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm \
            optuna shap kmodes umap-learn scipy statsmodels imbalanced-learn \
            scikit-learn-intelex numba
```

### 核心依赖包

```
pandas>=3.0.0
numpy>=1.22.0
scikit-learn>=1.5.0
xgboost>=3.0.0
lightgbm>=4.0.0
optuna>=4.0.0
shap>=0.45.0
kmodes>=0.12.2
umap-learn>=0.5.6
scikit-learn-intelex>=2025.0.0  # Intel CPU加速
torch>=2.5.1+cu121               # GPU支持（可选）
```

---

## 🚀 快速开始

### 方式一：运行Jupyter Notebooks

```bash
# （可选）开启Intel硬件加速，获得10-100倍性能提升
export SKLEARNEX_VERBOSE=INFO  # Linux/Mac
# 或 $env:SKLEARNEX_VERBOSE="INFO"  # Windows PowerShell

# 启动Jupyter Lab
jupyter lab

# 按顺序运行notebooks：
# 1. 01_Data_Understanding_EDA.ipynb
# 2. 02_Data_Preprocessing.ipynb
# 3. 03_Feature_Engineering.ipynb
# 4. 04_Clustering_K-Prototypes.ipynb
# 5. 05_Classification_Modeling.ipynb
```

**Intel加速提示**: 如果安装了`scikit-learn-intelex`，在notebook开头添加：
```python
from sklearnex import patch_sklearn
patch_sklearn()  # 自动加速scikit-learn算法
```

### 方式二：使用生产级推理脚本

```python
# 推荐方式：使用封装好的推理类
import sys
from pathlib import Path

# Get the directory where the current script is located
script_dir = Path(__file__).parent
model_dir = script_dir / "outputs" / "classification" / "models"

sys.path.append(str(model_dir))
from inference_pipeline import DiabetesRiskPredictorAtomic

# Initialize the predictor
predictor = DiabetesRiskPredictorAtomic(model_dir=str(model_dir))

# Sample patient data
sample_patient = {
    "HighBP": 1,
    "HighChol": 1,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "DiffWalk": 0,
    "Sex": 1,
    "Education": 5,
    "Income": 4,
    "MentHlth_Cat": 0,
    "PhysHlth_Cat": 1,
    "Age_BMI_Interaction": 280.5,
    "CVD_Risk": 2,
    "MetSyn_Risk": 3,
    "Chronic_Count": 2,
    "Lifestyle_Score": 3,
    "Risk_Behavior": 0,
    "BMI_Squared": 1056.25,
    "Health_Imbalance": 5,
    "Cluster_ID": 3,
    "Risk_Index": 32.94,
}

# Prediction
result = predictor.predict_risk(sample_patient)
print(f"Diabetes risk probability: {result['diabetes_risk']}")
print(f"Risk tier: {result['risk_tier']}")
print(
    f"Binary classification prediction (threshold={result['threshold_used']:.3f}): {'Positive' if result['prediction'] else 'Negative'}"
)
```

**注意**: 推理脚本已处理校准模型的兼容性问题，无需手动加载joblib文件。

---

## 📈 五阶段分析流程

### Phase 1: 数据理解与探索性分析 (EDA)

**Notebook**: [01_Data_Understanding_EDA.ipynb](01_Data_Understanding_EDA.ipynb)

**核心内容**:
- 数据完整性审计（零缺失值、23,889条重复记录分析）
- 目标变量分布分析（三分类 → 二分类转换依据）
- 特征-目标关联分析（Cramér's V、LOWESS平滑）
- BMI非线性风险升级验证（WHO临床分桶）
- 社会经济梯度评估（收入、教育与患病率）
- PCA结构评估（21个特征）
- 多重共线性检测（VIF分析）

**关键发现**:
- BMI≥30时糖尿病风险显著加速（非线性关系）
- `GenHlth`（主观健康）是最强预测因子（Cramér's V=0.380）
- 低收入群体患病率为高收入的2.4倍
- 前驱糖尿病标签存在诊断模糊性（建议合并为二分类）

---

### Phase 2: 数据预处理

**Notebook**: [02_Data_Preprocessing.ipynb](02_Data_Preprocessing.ipynb)

**核心内容**:
- **逻辑一致性清洗**: 移除416条矛盾记录（253,680→253,264）
  - 例：`CholCheck=0 & HighChol=1`（从未检测却高胆固醇）
- **Profile去重聚合**: 229,365条 → 229,296条唯一配置文件
  - 基于21个特征+目标变量的唯一组合
  - 保留`Sample_Weight`记录原始频次，加权总和=253,264
- **目标二值化**: `Diabetes_012` (0/1/2) → `Diabetes_binary` (0/1)，合并前驱糖尿病和糖尿病
- **权重标准化**: 归一化`Sample_Weight`均值=1.0，保持人群代表性

**输出数据集**:
- `CDC_Diabetes_Cleaned.csv`: 229,296条 × 23列
- 权重总和验证：253,264（移除416条矛盾记录后）

---

### Phase 3: 特征工程

**Notebook**: [03_Feature_Engineering.ipynb](03_Feature_Engineering.ipynb)

**核心内容**:
- **BMI异常值处理**: P99 Winsorization（BMI=98）保留极端信息
- **临床离散化**: 
  - `BMI_WHO` (低体重/正常/超重/肥胖) - WHO标准
  - `Age_Group` (13个年龄段，18-80+岁)
  - `MentHlth_Cat`, `PhysHlth_Cat` (健康天数分桶：0/1-13/14-29/30天)
- **交互特征合成**:
  - `Age_BMI_Interaction` = Age × BMI（协同效应捕获）
  - `CVD_Risk` = HighBP + HighChol + HeartDiseaseorAttack（心血管风险）
  - `MetSyn_Risk` = HighBP + HighChol + (BMI≥30)（代谢综合征）
- **聚合指数**:
  - `Chronic_Count`: 慢性疾病共病负担（0-5）
  - `Lifestyle_Score`: 保护性行为评分（0-4）
  - `Health_Imbalance`: |MentHlth - PhysHlth|（身心健康差异）
- **质量控制**: VIF筛选（阈值=10，移除高共线性特征）
- **数据分割**: 80/20分层划分（训练集183,436条，测试集45,860条）
- **特征标准化**: RobustScaler（中位数标准化，抗异常值）

**输出数据集**:
- 基线特征: `CDC_Train/Test_Classification_BASELINE.csv` (24个特征)
- 聚类特征: `CDC_Clustering_RAW/SCALED.csv` (21个特征，用于Phase 4)

---

### Phase 4: K-Prototypes聚类分析

**Notebook**: [04_Clustering_K-Prototypes.ipynb](04_Clustering_K-Prototypes.ipynb)

**核心内容**:
- **ReliefF特征选择**: 从21个特征中提取6个判别性特征（4连续+2分类）
- **Gamma参数校准**: 平衡连续和分类距离的权重（γ=4.274，自适应优化）
- **最优K值确定**: 三阶段层次策略
  1. 粗筛: K=2-10，评估聚类质量指标
  2. 细调: 45%小批量数据精细评估（K=2/3/6）
  3. 最终: **K=6**（Silhouette=0.313，Calinski_Harabasz=2912）
- **消融对比**: 
  - K-Means（仅连续）vs K-Modes（仅分类）vs K-Prototypes（混合）
  - 混合模型Silhouette Score优势: +14.3%
- **风险指数构建**: 基于患病率、共病负担、保护因子的0-100量化评分
- **社会公平性验证**: 聚类与收入/教育统计独立（χ² p>0.05）
- **Bootstrap稳定性**: ARI=0.83±0.02（500次重采样）
- **可解释性分析**: 
  - UMAP/t-SNE降维可视化
  - 雷达图展示6种风险表型
  - 全局代理决策树（深度=3，准确率81.23%）

**聚类画像**:

| Cluster | 规模 | 患病率 | 相对风险指数 | 风险分层 | 典型特征 |
|---------|------|--------|----------|----------|----------|
| **Cluster 4** | 36.3% | 3.4% | 2.62 | 低风险 | 年轻、正常BMI、无慢病 |
| **Cluster 2** | 5.1% | 5.8% | 4.98 | 中等（行为） | 重度饮酒、低体重 |
| **Cluster 0** | 24.9% | 17.9% | 15.68 | 高风险 | 久坐、中年、超重 |
| **Cluster 5** | 17.9% | 20.9% | 18.04 | 高风险 | 代谢综合征前期 |
| **Cluster 3** | 8.5% | 35.6% | 32.94 | 极高风险 | 心血管共病、高龄 |
| **Cluster 1** | 7.5% | 40.9% | 38.54 | 极高风险 | 多重慢病、肥胖 |

![聚类雷达图](outputs/clustering_k-prototypes/images/cluster_radar_profiles.png)
*图1: 六类风险表型的多维特征雷达图*

![UMAP降维可视化](outputs/clustering_k-prototypes/images/umap_manifold_exploration.png)
*图2: UMAP降维展示聚类分离度（颜色=风险指数，大小=样本密度）*

![聚类稳定性审计](outputs/clustering_k-prototypes/images/cluster_stability_audit.png)
*图3: Bootstrap稳定性审计展示高聚类可靠性（ARI=0.83±0.02，500次重采样）*

**输出数据集**:
- `CDC_Train/T[05_Classification_Modeling.ipynb](05_Classification_Modeling.ipynb)
- `CDC_Test/T[05_Classification_Modeling.ipynb](05_Classification_Modeling.ipynb)

### Phase 5: Classification 分类模型

**Notebook**: [05_Classification_Modeling.ipynb](05_Classification_Modeling.ipynb)

**核心内容**:
- **类别不平衡处理**:
  - 成本敏感学习: `scale_pos_weight=7.184` (FN成本:FP成本=5:1)
  - 样本加权: 使用`Sample_Weight`保持人群代表性
  - 类别不平衡比: 4.79:1（非糖尿病 vs 糖尿病）
- **特征消融实验**:
  - 基线特征（24个）vs 聚类增强（+Risk_Index/Cluster_ID，26个）
  - 聚类特征贡献: **+0.56%临床评分**、+0.08% AUC-ROC、+0.61% PR-AUC
- **多模型Benchmark**:
  - 6种算法: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, LightGBM
  - **冠军模型**: XGBoost Optimized（GPU hist加速）
  - 最佳性能: AUC-ROC **0.8193**, PR-AUC **0.4481**, Clinical Score **0.6296**,Recall **0.8508**
- **超参数优化**:
  - 框架: OPTUNA（贝叶斯TPE采样器）
  - 优化目标: PR-AUC（更适合不平衡数据）
  - 试验次数: 50次（5折分层交叉验证）
  - 最优参数: `max_depth=3`, `learning_rate=0.018`, `n_estimators=500`, `colsample_bytree=0.867`
- **概率校准**:
  - 方法: 等渗回归（Isotonic Regression，非参数）
  - 改善: Brier Score ↓ **0.103** (从0.211降至0.108)
  - ECE=**0.0068**（期望校准误差，接近完美）
- **阈值优化**:
  - 约登指数阈值: **0.147**（推荐，最大化灵敏度+特异性）
  - 高特异性阈值: 0.332（90%特异性，适用于筛查场景）
  - 默认阈值: 0.5（高精确度，但灵敏度仅16%）
- **全面评估**:
  - ROC/PR曲线（Bootstrap 95% CI，1000次重采样）
  - 多阈值混淆矩阵（5个临床场景）
  - 校准曲线（可靠性图，10分位）
  - Brier Score分解（Resolution/Reliability/Uncertainty）
- **可解释AI**:
  - SHAP全局特征重要性（TreeExplainer，2000个背景样本）
  - SHAP依赖图（双向交互效应可视化）
  - Force Plot（个案决策路径）: TP/FP/FN典型案例解析
- **公平性审计**:
  - 按性别分层: 男性AUC 0.815，女性AUC 0.823（差异<1%）
  - 按收入分层: 低收入群体Recall保持80%+，无显著歧视

![ROC-PR曲线](outputs/classification/images/evaluation_roc_pr_curves.png)
*图4: 模型性能评估 - ROC曲线（左）和PR曲线（右）含Bootstrap 95%置信区间*

![SHAP特征重要性](outputs/classification/images/shap_feature_importance_bar.png)
*图5: SHAP全局特征重要性排名（Top 15）*

**模型部署**:
- 核心模型文件:
  - `champion_model_calibrated.pkl`: 校准后的最优XGBoost模型（推荐使用）
  - `champion_model_uncalibrated.pkl`: 未校准的原始模型
  - `XGBClassifier_Optimized_champion.pkl`: OPTUNA优化后的模型
- 配置文件: `feature_configuration.json`（26个特征顺序）
- 推理脚本: `inference_pipeline.py`（DiabetesRiskPredictorAtomic类）

---

### 📏 评估指标说明

#### Clinical Score（临床综合评分）

**定义**: `Clinical Score = 0.6 × Recall + 0.4 × Precision`

**设计理念**:
- **Recall（灵敏度/召回率）**: 糖尿病**阳性类**的召回率，衡量模型正确识别真实患者的能力
  - 计算公式: TP / (TP + FN)
  - 临床意义: **漏诊率 = 1 - Recall**，Recall越高，漏诊越少
  
- **Precision（精确度）**: 糖尿病**阳性类**的精确度，衡量阳性预测的准确性
  - 计算公式: TP / (TP + FP)
  - 临床意义: **误报率 = 1 - Precision**，Precision越高，假阳性越少

**权重分配原因**:
1. **公共卫生优先原则**: 糖尿病早期干预成本低、效果好，**漏诊的代价远高于误报**
2. **疾病严重性**: 未确诊的糖尿病患者可能发展为严重并发症（失明、肾衰竭、截肢），而假阳性仅需复检确认
3. **筛查场景适配**: 60%权重给Recall，确保**高灵敏度**，适用于社区初筛
4. **经济学考量**: 基于DPP研究，每预防1例糖尿病节省医疗费用$8,800，假阳性复检成本仅$50-100

**对比F1-Score**:
- F1 = 2 × (Precision × Recall) / (Precision + Recall)，给予两者**相等权重**
- Clinical Score给予Recall **1.5倍权重**（0.6 vs 0.4），更符合临床决策优先级

**本项目表现**:
- Clinical Score: **0.6296**（聚类增强）vs 0.6240（基线）
- Recall: 80.21%（高灵敏度，仅漏诊19.79%）
- Precision: 32.28%（可接受的误报率，通过复检过滤）

---

**测试集评估** (n=45,860，约登指数阈值=0.147):

| 指标 | 数值 | 95% CI | 临床意义 |
|------|------|--------|----------|
| **AUC-ROC** | **0.8191** | [0.814, 0.824] | 优秀的整体判别能力 |
| **PR-AUC** | **0.4390** | [0.433, 0.445] | 不平衡数据下的真实性能 |
| **Recall (灵敏度)** | **80.21%** | [79.2%, 81.2%] | 正确识别糖尿病患者比例 |
| **Precision (精确度)** | **32.28%** | [31.5%, 33.1%] | 阳性预测准确率 |
| **Specificity (特异性)** | **68.35%** | [67.9%, 68.8%] | 正确识别健康人比例 |
| **F1 Score** | **0.4604** | [0.456, 0.465] | 平衡准确率和召回率 |
| **Brier Score** | **0.1076** | [0.105, 0.110] | 概率预测准确性（越低越好） |
| **ECE** | **0.0068** | - | 期望校准误差（接近完美） |

**多阈值对比** (灵敏度-特异性权衡):

| 阈值策略 | 阈值 | 灵敏度 | 特异性 | PPV | NPV | 临床场景 |
|---------|------|--------|--------|-----|-----|----------|
| 高灵敏度 | 0.30 | 56.6% | 85.2% | 41.9% | 91.3% | 社区筛查（减少漏诊） |
| **约登指数**⭐ | **0.147** | **80.2%** | **68.4%** | **32.3%** | **94.8%** | **推荐使用（平衡最优）** |
| 高特异性 | 0.332 | 46.2% | 89.7% | 45.7% | 89.9% | 医保控费（减少误诊） |
| 默认阈值 | 0.50 | 16.4% | 97.7% | 57.8% | 86.1% | 高精度确诊（漏诊率高） |

> **临床解读**: 
> - **约登指数阈值（0.147）推荐用于一般筛查**: 在45,860名测试者中，可正确识别5,802名真实糖尿病患者（80.2%），同时会将12,171名健康人误判（可接受的过度筛查代价，NPV高达94.8%说明阴性预测非常可靠）
> - **高特异性阈值（0.332）适用于资源受限场景**: 减少61%假阳性（降至3,970人），但会漏诊38.9%真实患者

![校准曲线](outputs/classification/images/calibration_uncalibrated_reliability.png)
*图6: 概率校准前后对比（等渗回归显著改善预测可靠性）*

### Top 10 SHAP特征重要性

基于TreeExplainer对测试集45,860样本的全局归因分析：

| 排名 | 特征 | SHAP均值 | 临床意义 | 特征类型 |
|------|------|----------|----------|----------|
| 1 | **Age_BMI_Interaction** | 0.4150 | 年龄与肥胖的协同效应（乘法项） | 交互特征 |
| 2 | **GenHlth** | 0.3529 | 主观健康评分（1=优秀 → 5=差） | 原始特征 |
| 3 | **Risk_Index** | 0.3297 | 聚类风险量化评分（0-100） | 聚类衍生 |
| 4 | **MetSyn_Risk** | 0.2592 | 代谢综合征风险因子数（0-3） | 聚合特征 |
| 5 | **Chronic_Count** | 0.2538 | 慢性疾病共病负担（0-5） | 聚合特征 |
| 6 | **BMI_Squared** | 0.1213 | BMI平方项（非线性风险捕获） | 工程特征 |
| 7 | **Income** | 0.1160 | 年收入等级（1-8，社会经济地位） | 原始特征 |
| 8 | **Sex** | 0.0920 | 性别（0=女，1=男，男性风险↑） | 原始特征 |
| 9 | **HighBP** | 0.0429 | 高血压诊断（0=否，1=是） | 原始特征 |
| 10 | **DiffWalk** | 0.0371 | 行走困难（功能性残障） | 原始特征 |

> **关键洞察**: 前5名特征贡献了60%的模型解释力，其中**Risk_Index（聚类特征）排名第3**，验证了无监督预处理的价值。

### SHAP依赖图深度解析

![SHAP依赖图](outputs/classification/images/shap_dependence_plots.png)
*图7: Top 4特征的SHAP依赖图（颜色=交互特征值，展示双向效应）*

**核心发现**:

1. **Age_BMI_Interaction** (SHAP=0.415):
   - **非线性陡增**: 值>300时风险急剧上升（对应60岁+肥胖）
   - **交互效应**: GenHlth（颜色）放大影响，健康不佳者效应翻倍
   - **临床阈值**: 250-350为关键干预窗口期

2. **GenHlth** (SHAP=0.353):
   - **阶跃模式**: 评分≥4（健康不佳/差）触发强烈风险信号
   - **协同作用**: 与Chronic_Count（颜色）正相关，多重慢病进一步恶化主观感受
   - **早期预警**: 评分从3→4的转变是高风险预警信号

3. **Risk_Index** (SHAP=0.330):
   - **S型曲线**: 30-40分段风险陡增（对应Cluster 3/1极高风险组）
   - **验证聚类**: 与真实糖尿病状态高度一致，Spearman相关系数0.68
   - **分层明确**: <10分（低风险），10-25分（中风险），>30分（高风险）

4. **MetSyn_Risk** (SHAP=0.259):
   - **累加效应**: 每增加1个代谢因子，SHAP值上升0.15
   - **阈值识别**: ≥2个因子（高血压+高血脂+肥胖）为代谢综合征警戒线
   - **性别差异**: 男性（颜色深）在相同风险因子下SHAP值更高

### 个案决策路径分析（Force Plot）

![Force Plot案例](outputs/classification/images/shap_force_plot_true_positive_high_risk.png)
*图8: 真阳性案例的SHAP Force Plot（基础风险0.21→预测0.87，主要驱动因子标注）*

---

## 🏥 公共卫生应用

### 基于聚类的精准干预策略

根据K-Prototypes聚类识别的6类风险表型，制定差异化公共卫生干预方案：

![干预策略全景](outputs/clustering_k-prototypes/images/cluster_archetype_heatmap.png)
*图9: 六类人群的多维特征热力图（标准化z-score，红=高于均值，蓝=低于均值）*

#### 🔴 一级预警：极高风险群体（Cluster 1 & 3）

| 指标 | Cluster 1 | Cluster 3 |
|------|-----------|-----------|
| **规模** | 7.5% | 8.5% |
| **患病率** | 40.9% | 35.6% |
| **风险指数** | 38.54 | 32.94 |
| **核心特征** | 多重慢病+肥胖 | 心血管共病+高龄 |

**临床画像**:
- **Cluster 1**: 平均BMI 38.5（重度肥胖），GenHlth 4.2（健康不佳），慢性病数≥3
- **Cluster 3**: 平均年龄68岁，心脏病22%，高血压84%，中风史9%

**集约化干预措施**:
1. **临床管理**:
   - 每季度代谢指标监测（HbA1c、血脂、肾功能、血压）
   - 心血管-代谢联合管理（双重保护：心脏+肾脏）
   - 药物治疗评估（他汀类、SGLT-2抑制剂、GLP-1受体激动剂）
   
2. **专科协作**:
   - 内分泌科+心内科联合会诊
   - 快速转诊通道（眼底筛查、神经病变检测）
   
3. **成本效益**: 集中资源于15.9%人口，预防40%的糖尿病发病

**预期效果**: 通过密集管理，5年内糖尿病进展风险降低30%（基于DPP研究）

---

#### 🟠 二级关注：高风险群体（Cluster 0 & 5）

| 指标 | Cluster 0 | Cluster 5 |
|------|-----------|-----------|
| **规模** | 26.8% | 18.6% |
| **患病率** | 17.9% | 21.0% |
| **风险指数** | 15.68 | 18.04 |
| **核心特征** | 久坐+超重 | 代谢综合征前期 |

**临床画像**:
- **Cluster 0**: 平均BMI 28.3（超重），体力活动率38%，中年（45-60岁）
- **Cluster 5**: 高血压+高血脂双达标率仅12%，腹型肥胖76%

**生活方式干预方案**:
1. **行为矫正**:
   - **运动处方**: 150分钟/周中等强度有氧运动（快走、游泳、骑行）
   - **医学营养治疗（MNT）**: 
     - 热量赤字500 kcal/天
     - 限制精制糖和饱和脂肪
     - 增加全谷物和膳食纤维（25-30g/天）
   
2. **体重管理**:
   - **目标**: 6个月内减重5-7%（基于DPP研究最佳阈值）
   - **监测**: 每月生物测量（体重、腰围、血压）
   
3. **社区支持**:
   - 群组行为改变计划（16周课程）
   - 移动健康APP提醒和追踪
   - 每年生化筛查（空腹血糖、HbA1c）

**预期效果**: 3年内糖尿病发病率降低58%（DPP研究数据）

---

#### 🟡 三级监测：中等风险群体（Cluster 2）

**规模**: 5.5% | **患病率**: 5.8% | **风险指数**: 4.98

**临床画像**: 
- 重度饮酒100%（定义：男性≥14杯/周，女性≥7杯/周）
- 平均BMI 22.8（偏低），可能存在营养不良
- GenHlth 2.9（自评健康较好，但行为风险高）

**行为风险缓解策略**:
1. **酒精干预**:
   - 短暂干预疗法（SBIRT，初级保健整合）
   - 目标：减量至低风险水平（男性<14杯/周）
   
2. **代谢监测**:
   - 肝功能检测（ALT、AST、GGT）
   - 甘油三酯监测（酒精性高甘油三酯血症）
   - 胰腺功能评估（酒精性胰腺炎风险）
   
3. **营养支持**:
   - 维生素B1补充（硫胺素缺乏症预防）
   - 蛋白质营养评估（防止肌少症）

**特殊考虑**: 这是唯一具有显著行为风险因素的聚类，需心理健康筛查（抑郁/焦虑共病率高）

---

#### 🟢 四级维护：低风险群体（Cluster 4）

**规模**: 31.5% | **患病率**: 3.4% | **风险指数**: 2.62

**临床画像**:
- 平均年龄38岁，BMI 24.6（正常范围）
- 体力活动率82%，慢性病率<5%
- 健康行为评分高（不吸烟、适量饮酒、均衡饮食）

**预防维护策略**:
1. **人群监测**:
   - 年度健康风险评估（HRA）
   - 生物测量追踪（BMI、血压、血糖）
   
2. **健康促进**:
   - 数字化健康素养课程（糖尿病预防知识）
   - 社区健康讲座（代谢健康、运动营养）
   - 工作场所健康计划（站立办公、健康餐饮）
   
3. **激励机制**:
   - 健康行为积分奖励
<div align="center">

**DiaMetric-CDC: Diabetes Risk Prediction System**

基于2015 CDC BRFSS数据 | 混合建模范式 | 临床级可解释性

---

*本项目展示了从数据理解到生产部署的完整机器学习工程实践*

```
高优先级 (Critical) → Cluster 1/3（集约型临床管理）
中优先级 (High)     → Cluster 0/5（生活方式干预）
低优先级 (Moderate) → Cluster 2（行为咨询）
维持级别 (Low)      → Cluster 4（预防宣教）
```

---

## 💡 技术亮点

### 1. 混合建模范式

- **无监督前置**: K-Prototypes聚类发现潜在风险表型
- **监督精调**: XGBoost学习聚类增强特征
- **协同增效**: 聚类Risk_Index成为第3重要特征（SHAP=0.330）

### 2. 高性能计算优化

- **Intel Extension for Scikit-learn**: CPU算法加速（K-Means/KNN）
- **XGBoost GPU hist**: CUDA加速决策树训练
- **NUMBA JIT**: ReliefF特征选择并行化
- **训练时间**: Phase 5完整流程<30分钟（含超参数优化）

### 3. 概率校准工程

- **校准方法**: 等渗回归（非参数单调映射）
- **评估指标**: ECE（期望校准误差）= 0.007
- **实用价值**: 预测概率可直接用于风险沟通（"您有32%的糖尿病风险"）

### 4. 严格的统计验证

- **Bootstrap CI**: 1000次重采样估计性能区间
- **配对t检验**: 消融实验统计显著性
- **分层交叉验证**: 5-fold CV保持类别平衡
- **稳定性审计**: 聚类ARI=0.83±0.02（高稳定性，通过500次Bootstrap重采样验证）

### 5. 鲁棒工程实践

- **原子级推理回退机制**: 实现了基于属性探测的多层Fallback策略
  - 优先使用`CalibratedClassifierCV`的校准概率
  - 校准层失效时自动降级至`base_estimator`原始预测
  - 确保在边缘情况下仍能提供可靠结果（解决NaN兼容性问题）
- **Intel硬件加速**: 集成`scikit-learn-intelex`，K-Means/KNN算法加速10-100倍
- **GPU加速训练**: XGBoost使用`device='cuda'`和`tree_method='hist'`
- **模块化推理类**: `DiabetesRiskPredictorAtomic`封装完整预测流程，支持单样本/批量推理

### 6. 端到端可重现性

- **固定随机种子**: `RANDOM_STATE=42`（贯穿所有阶段）
- **版本锁定**: `requirements.txt`指定所有依赖版本（含CUDA版本）
- **数据血缘**: 完整追踪从原始数据到模型输出的每一步转换
- **元数据管理**: JSON格式保存所有配置和结果（超参数、特征顺序、性能指标）

---

## ⚠️ 模型局限性与批判性思考

### 1. 数据源局限性

#### 截面偏差 (Cross-Sectional Bias)
- **问题**: BRFSS 2015为截面调查，无法建立**因果关系**
- **影响**: 例如"HighBP→糖尿病"可能是双向关联或共同原因（如肥胖）
- **缓解策略**: 
  - 未来应整合2016-2026年纵向数据，进行生存分析 (Cox回归)
  - 使用工具变量法 (Instrumental Variables) 评估干预效果

#### 自报偏差 (Self-Report Bias)
- **问题**: 21个特征均为主观自述，缺少客观生物标志物 (HbA1c/空腹血糖)
- **风险**: 
  - BMI可能被低估 (社会期望偏差)
  - 糖尿病患者中未诊断比例高达20-25% (CDC估计)
- **验证需求**: 未来应与NHANES (National Health and Nutrition Examination Survey) 客观检测数据交叉验证

#### 幸存者偏差 (Survivorship Bias)
- **问题**: 严重并发症患者可能因病死/无法参与调查而缺失
- **后果**: 模型可能**低估极端风险**，实际致病率高于训练数据

---

### 2. 算法选择的权衡

#### K-Prototypes 的假设限制
- **Gamma参数敏感性**: 连续/分类特征权重平衡 (γ=4.274) 需领域专家验证
- **球形聚类假设**: 可能无法捕捉非凸/任意形状聚类
- **替代方案**: 
  - 尝试密度聚类 (DBSCAN) 或混合高斯模型 (GMM)
  - 使用UMAP降维可视化验证聚类分离度

![UMAP聚类可视化](outputs/clustering_k-prototypes/images/umap_manifold_exploration.png)
*图10: UMAP流形降维展示聚类分离性（颜色=聚类标签，可见部分重叠区域）*

#### XGBoost 的黑盒风险
- **可解释性与性能的拉锯战**: 尽管有SHAP，但500棵决策树仍难以向临床医生直观解释
- **过拟合风险**: 尽管设置max_depth=3，但500棵树可能记忆训练集噪声
- **透明度措施**:
  - 使用全局代理模型 (Surrogate Decision Tree, 深度=4, 准确率86%)
  - 提供个案Force Plot，展示具体预测逻辑

![代理决策树](outputs/clustering_k-prototypes/images/surrogate_decision_tree_refined.png)
*图11: 全局代理决策树（深度=4，86%准确率模拟K-Prototypes聚类逻辑）*

---

### 3. 性能指标的临床意义

#### Precision-Recall 不平衡
- **PR-AUC=0.439**: 在高度不平衡数据下，此值**仅为基准线2.8倍** (Random: 0.158)
- **Precision=32.3%**: 意味着**68%的阳性预测为误报**，可能导致过度医疗化
- **临床权衡**: 
  - 社区筛查: 接受68%误报率，换取Recall=80.2%（减少漏诊）
  - 医保控费: 需提高阈值至0.332，精确度↑而灵敏度↓

#### 校准的局限性
- **ECE=0.007并非完美**: 等渗回归只能保证**训练集分布**下的校准，新人群可能漂移
- **建议**: 部署后应定期监控预测概率与实际结果的匹配度 (Calibration Drift)

---

### 4. 公平性与伦理问题

#### 群体公平性分析
![公平性审计](outputs/classification/images/subgroup_fairness_analysis.png)
*图12: 按性别和收入分层的模型性能审计（AUC差异<1%，符合公平性标准）*

- **收入分层**: 低收入群体Recall保持80%+，但Precision可能更低
- **风险**: 低社会经济地位人群可能承担更多**假阳性心理负担**
- **缓解**: 建议对低收入群体提供**免费复筛**（HbA1c检测）

#### 算法歧视风险
- **问题**: 模型可能强化现有健康不平等 (例如将低收入与高风险关联)
- **伦理挑战**: 如果保险公司使用此模型，可能导致**费率歧视**
- **解决方案**:
  - 移除敏感特征 (Income/Education) 后重新训练
  - 使用**公平性约束算法** (Fairness-constrained Learning)

---

### 5. 泛化性挑战

#### 时间漂移 (Temporal Drift)
- **问题**: 2015年数据可能无法适用于2026年人群 (生活方式变化/医疗技术进步)
- **验证**: 未在2016-2026年BRFSS数据上测试模型性能
- **建议**: 部署后实时监控性能退化，每年使用新数据重训

#### 人群差异 (Population Shift)
- **BRFSS特殊性**: 美国成年人电话调查，**不适用于其他国家**
- **风险**: 亚洲/非洲人群的BMI-糖尿病关系不同（亚洲人更低BMI即高风险）
- **跨文化验证需求**: 在中国慧人队列/英国Biobank数据上验证泛化性

![行为风险梯度](outputs/clustering_k-prototypes/images/behavioral_risk_gradient_integrated.png)
*图13: 行为风险因子梯度分析（保护性行为越多，风险越低）*

---

### 6. 消融实验的批判性视角

![消融实验结果](outputs/classification/images/ablation_confusion_matrix_comparison.png)
*图14: 消融实验混淆矩阵对比（基线vs聚类增强，聚类特征仅贡献+0.56%临床评分）*

- **边际改善**: 聚类特征（Risk_Index/Cluster_ID）提升指标：
  - Clinical Score: **+0.56%** (主要指标，从0.5368→0.5398)
  - AUC-ROC: **+0.08%** (从0.8005→0.8011，统计显著p<0.001)
  - PR-AUC: **+0.61%** (从0.4210→0.4236)
- **成本-收益**: 聚类阶段耗时~10分钟，是否值得在生产环境中保留？
- **建议**: 对资源受限场景，**基线XGBoost模型已足够** (AUC=0.8005，默认参数)
  - **注**: 超参优化（OPTUNA）将AUC提升至**0.8191**，远高于聚类特征贡献

![消融交叉验证](outputs/classification/images/ablation_cv_distributions.png)
*图15: 5折交叉验证消融实验分布（显示聚类增强的稳定改善）*

---

### 7. 实施建议

#### 部署前必需步骤
1. **临床验证**: 在小范围前瞻性研究中验证预测与实际结果的一致性
2. **医学伦理审查**: IRB (Institutional Review Board) 批准
3. **HIPAA合规**: 确保患者隐私保护
4. **医生培训**: 教育临床人员正确解读预测结果

#### 持续监控框架
- **性能退化监控**: 每季度评估AUC/Recall是否下降
- **公平性监控**: 按种族/性别/收入分层评估性能差异
- **反馈循环**: 收集临床医生反馈，迭代优化模型

---

## 🚧 未来展望

### 短期改进

- [ ] **时序扩展**: 整合2016-2026年BRFSS数据追踪趋势变化
- [ ] **增量学习**: 实现在线学习框架适应新数据
- [ ] **特征增强**: 纳入遗传风险评分（GRS）和饮食质量指数
- [ ] **模型集成**: Stacking融合XGBoost/LightGBM/CatBoost

### 中期规划

- [ ] **因果推断**: 使用工具变量法评估干预效果
- [ ] **生存分析**: Cox回归预测糖尿病发病时间（time-to-event）
- [ ] **多任务学习**: 同时预测糖尿病/心血管疾病/慢性肾病
- [ ] **联邦学习**: 跨医疗机构协作建模（隐私保护）

### 长期愿景

- [ ] **临床决策支持系统（CDSS）**: 嵌入电子健康记录（EHR）
- [ ] **移动健康应用**: 个性化风险评估和行为干预APP
- [ ] **政策模拟**: 评估不同公共卫生干预的成本效益
- [ ] **全球健康**: 跨国数据集验证模型泛化性



---

<div align="center">

**DiaMetric-CDC: 糖尿病风险预测系统**

基于2015 CDC BRFSS数据 | 混合建模范式 | 临床级可解释性

---

*本项目展示了从数据理解到生产部署的完整机器学习工程实践*

[⬆ 回到顶部](#diametric-cdc-基于cdc-brfss数据的糖尿病风险智能预测系统)

</div>
