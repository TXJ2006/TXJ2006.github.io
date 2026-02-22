---
title: "上册：机器学习与现代架构"
layout: "single"
url: "/book/volume1/"
summary: "The Mathematical Phenomenology of Machine Intelligence — Volume I: Machine Learning & Modern Architectures"
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
---

<div style="text-align:center; margin: 1.5em 0;">

# 上册：机器学习与现代架构

### Volume I: Machine Learning & Modern Architectures

*共 6 卷 · 19 Parts · 200 章*

</div>

---

## 第一卷：数学基底与公理化 {#vol1}
### Volume I: The Mathematical Substrate

> *定义语言。如果不理解希尔伯特空间和测度论，就无法理解现代 AI。*

---

### Part A: 线性代数与泛函分析 {#part-a}
*Linear Algebra & Functional Analysis*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 001 | [向量空间与对偶空间](/book/chapters/chapter001/) | Vector Spaces & Dual Spaces |
| 002 | [谱理论：特征值与奇异值分解](/book/chapters/chapter002/) | Spectral Theory: Eigendecomposition & SVD |
| 003 | [张量代数与爱因斯坦求和约定](/book/chapters/chapter003/) | Tensor Algebra & Einstein Summation |
| 004 | [希尔伯特空间与范数](/book/chapters/chapter004/) | Hilbert Spaces & Norms |
| 005 | [巴拿赫空间与算子范数](/book/chapters/chapter005/) | Banach Spaces & Operator Norms |
| 006 | [里斯表示定理](/book/chapters/chapter006/) | Riesz Representation Theorem |
| 007 | [再生核希尔伯特空间 (RKHS) — 核方法的灵魂](/book/chapters/chapter007/) | Reproducing Kernel Hilbert Spaces |
| 008 | [矩阵微积分](/book/chapters/chapter008/) | Matrix Calculus |
| 009 | [随机矩阵理论 — 理解高维初始化的关键](/book/chapters/chapter009/) | Random Matrix Theory |
| 010 | [凸集与凸锥](/book/chapters/chapter010/) | Convex Sets & Cones |

---

### Part B: 概率论与测度 {#part-b}
*Probability & Measure*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 011 | [$\sigma$-代数与可测空间](/book/chapters/chapter011/) | $\sigma$-Algebras & Measurable Spaces |
| 012 | [信息论与熵 — 对数泛函与学习即压缩](/book/chapters/chapter012/) | Information Theory & Entropy |
| 013 | [拉东-尼科迪姆导数 — 密度比的本质](/book/chapters/chapter013/) | Radon-Nikodym Derivative |
| 014 | [条件期望作为正交投影](/book/chapters/chapter014/) | Conditional Expectation as Orthogonal Projection |
| 015 | [马尔可夫链与遍历性](/book/chapters/chapter015/) | Markov Chains & Ergodicity |
| 016 | [鞅论 — 强化学习收敛性的基础](/book/chapters/chapter016/) | Martingale Theory |
| 017 | [大数定律与中心极限定理](/book/chapters/chapter017/) | LLN & CLT |
| 018 | [集中不等式 — 学习理论的界](/book/chapters/chapter018/) | Concentration Inequalities: Hoeffding/Bernstein |
| 019 | [随机微分方程 (SDE) 基础 — 扩散模型的基础](/book/chapters/chapter019/) | Itô Calculus |
| 020 | [维纳过程与布朗运动](/book/chapters/chapter020/) | Wiener Processes |

---

### Part C: 信息论与几何 {#part-c}
*Information & Geometry*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 021 | [香农熵与互信息](/book/chapters/chapter021/) | Shannon Entropy & Mutual Information |
| 022 | KL 散度与相对熵 | Kullback-Leibler Divergence |
| 023 | 最大熵原理 | The Maximum Entropy Principle |
| 024 | 费雪信息矩阵 — 自然梯度的基础 | Fisher Information Matrix |
| 025 | 黎曼流形与度量张量 | Riemannian Manifolds & Metric Tensors |
| 026 | 测地线与指数映射 | Geodesics & Exponential Maps |
| 027 | 平行移动与协变导数 | Parallel Transport & Covariant Derivatives |
| 028 | 曲率张量与截面曲率 | Curvature Tensors |
| 029 | 辛几何与哈密顿系统 | Symplectic Geometry |
| 030 | 最优传输与 Wasserstein 距离 | Optimal Transport & Wasserstein Distance |

---

## 第二卷：经典统计学习 {#vol2}
### Volume II: Classical Statistical Learning

> *理解"小数据"时代的智慧。这些模型有严格的数学保证。*

---

### Part D: 频率派与贝叶斯派 {#part-d}
*Frequentist & Bayesian*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 031 | 极大似然估计 (MLE) 的渐近正态性 | MLE Asymptotic Normality |
| 032 | 贝叶斯推断与后验分布 | Bayesian Inference |
| 033 | 共轭先验与指数族分布 | Conjugate Priors & Exponential Families |
| 034 | 期望最大化 (EM) 算法 — 潜变量模型的鼻祖 | Expectation-Maximization |
| 035 | 变分推断 — 近似贝叶斯的基石 | Variational Inference |
| 036 | 蒙特卡洛马尔可夫链 (MCMC) | Gibbs & Metropolis-Hastings |
| 037 | 非参数统计与核密度估计 | Kernel Density Estimation |
| 038 | 广义线性模型 | GLM |
| 039 | 结构风险最小化 | Structural Risk Minimization |
| 040 | AIC, BIC 与模型选择准则 | Model Selection Criteria |

---

### Part E: 浅层学习架构 {#part-e}
*Shallow Architectures*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 041 | 线性回归的几何意义 | Geometry of Least Squares |
| 042 | 岭回归与 Lasso ($L^1$ vs $L^2$) | Ridge & Lasso |
| 043 | 逻辑回归与感知机 | Logistic Regression & Perceptrons |
| 044 | 支持向量机 (SVM) 与对偶优化 | SVM & Dual Optimization |
| 045 | 核技巧与 Mercer 定理 | The Kernel Trick & Mercer's Theorem |
| 046 | 决策树与信息增益 | Decision Trees & CART |
| 047 | 集成学习：Bagging 与 Random Forest | Ensemble: Bagging & RF |
| 048 | 提升算法：AdaBoost 与 Gradient Boosting | AdaBoost & XGBoost |
| 049 | 主成分分析 (PCA) 与 SVD 的关系 | PCA & SVD |
| 050 | 独立成分分析 (ICA) 与盲源分离 | ICA & Blind Source Separation |
| 051 | 线性判别分析 (LDA) | Linear Discriminant Analysis |
| 052 | 典型相关分析 (CCA) | Canonical Correlation Analysis |
| 053 | t-SNE 与流形降维 | Manifold Dimensionality Reduction |
| 054 | k-Means 与高斯混合模型 (GMM) | k-Means & GMM |
| 055 | 谱聚类 — 图拉普拉斯算子 | Spectral Clustering |

---

### Part F: 概率图模型 {#part-f}
*Probabilistic Graphical Models*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 056 | 贝叶斯网络 | Bayesian Networks |
| 057 | 马尔可夫随机场 (MRF) | Markov Random Fields |
| 058 | 因子图与和积算法 | Factor Graphs & Sum-Product |
| 059 | 隐马尔可夫模型 (HMM) | Hidden Markov Models |
| 060 | 条件随机场 (CRF) | Conditional Random Fields |

---

## 第三卷：深度学习与表示 {#vol3}
### Volume III: Deep Learning & Representation

> *从手工特征到自动特征提取。理解网络的结构对称性。*

---

### Part G: 前馈与优化 {#part-g}
*Feedforward & Optimization*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 061 | 多层感知机 (MLP) 与通用近似定理 | Universal Approximation |
| 062 | 反向传播的链式法则本质 | Backpropagation |
| 063 | 随机梯度下降 (SGD) 的动力学 | SGD Dynamics |
| 064 | 动量法与 Nesterov 加速 | Momentum & Nesterov |
| 065 | 自适应优化器的黎曼几何解释 | Adam, RMSProp |
| 066 | 激活函数的选择逻辑 | ReLU, GeLU, Swish |
| 067 | 权重初始化 | Xavier & Kaiming |
| 068 | 批归一化的协变量偏移理论 | BatchNorm |
| 069 | 层归一化与 RMSNorm | LayerNorm & RMSNorm |
| 070 | Dropout 作为贝叶斯近似 | Dropout as Bayesian |
| 071 | 权重衰减与 $L^2$ 正则化的解耦 | Weight Decay vs L2 |
| 072 | 梯度裁剪与数值稳定性 | Gradient Clipping |
| 073 | 学习率调度策略 | Warmup & Cosine Decay |
| 074 | 早停法作为隐式正则化 | Early Stopping |
| 075 | 神经切线核 (NTK) 理论 | Neural Tangent Kernel |

---

### Part H: 结构与归纳偏置 {#part-h}
*Structure & Inductive Bias*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 076 | 卷积神经网络 (CNN) 与平移等变性 | CNN & Translation Equivariance |
| 077 | 扩张卷积与感受野 | Dilated Convolution |
| 078 | 残差网络 (ResNet) 与欧拉离散化 ODE | ResNet & Euler ODE |
| 079 | DenseNet 与特征复用 | DenseNet |
| 080 | U-Net 架构与多尺度特征融合 | U-Net |
| 081 | 循环神经网络 (RNN) 与 BPTT | RNN & BPTT |
| 082 | LSTM/GRU 的门控机制与梯度流 | LSTM/GRU |
| 083 | 双向 RNN 与序列标注 | Bidirectional RNN |
| 084 | 序列到序列与编码器-解码器 | Seq2Seq |
| 085 | 注意力机制作为核平滑 | Attention as Kernel Smoothing |
| 086 | Transformer 架构详解 | Transformer |
| 087 | 自注意力的二次复杂度分析 | Self-Attention Complexity |
| 088 | 位置编码 (RoPE, ALiBi) | Positional Encodings |
| 089 | 视觉 Transformer (ViT) | ViT & Patch Embedding |
| 090 | 混合专家模型 (MoE) 与稀疏路由 | Mixture of Experts |

---

### Part I: 图与几何深度学习 {#part-i}
*Geometric Deep Learning*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 091 | 图神经网络 (GNN) 基础 | GNN Foundations |
| 092 | 消息传递神经网络 (MPNN) | Message Passing |
| 093 | 图卷积网络的谱域解释 | GCN Spectral |
| 094 | 图注意力网络 (GAT) | Graph Attention |
| 095 | 等变神经网络 E(n) | Equivariant Networks |
| 096 | 3D 点云处理 (PointNet) | PointNet |
| 097 | 神经隐式表示 (INR / SIREN) | Neural Implicit Representations |
| 098 | 神经辐射场 (NeRF) | NeRF |
| 099 | 高斯泼溅 (3D Gaussian Splatting) | 3DGS |
| 100 | 流形学习与自编码器 | Autoencoders |

---

## 第四卷：生成式 AI 与概率建模 {#vol4}
### Volume IV: Generative AI

> *从判别式模型（预测标签）转向生成式模型（预测分布）。*

---

### Part J: 显式密度模型 {#part-j}
*Explicit Density Models*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 101 | 变分自编码器 (VAE) 详解 | VAE |
| 102 | 重参数化技巧 | Reparameterization Trick |
| 103 | VAE 的后验崩塌问题 | Posterior Collapse |
| 104 | 向量量化 VAE (VQ-VAE) | VQ-VAE |
| 105 | 归一化流 | Normalizing Flows |
| 106 | 自回归模型 (PixelCNN, WaveNet) | Autoregressive Models |
| 107 | 掩码语言模型 (BERT) | Masked Language Model |
| 108 | 因果语言模型 (GPT) | Causal Language Model |

---

### Part K: 隐式与得分模型 {#part-k}
*Implicit & Score Models*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 109 | 生成对抗网络 (GAN) 基础 | GAN Foundations |
| 110 | GAN 的纳什均衡与极小极大博弈 | Nash Equilibrium & Minimax |
| 111 | 模式崩塌的原因 | Mode Collapse |
| 112 | Wasserstein GAN 与 Lipschitz 约束 | WGAN |
| 113 | 扩散模型 (DDPM) 的热力学推导 | Diffusion Models |
| 114 | SDE 视角的扩散 | Score-based Generative Modeling |
| 115 | 朗之万动力学采样 | Langevin Dynamics |
| 116 | 无分类器引导 | Classifier-Free Guidance |
| 117 | 潜在扩散模型 (Stable Diffusion) | Latent Diffusion |
| 118 | 一致性模型 | Consistency Models |
| 119 | 流匹配 | Flow Matching |
| 120 | 视频生成模型 (Sora) | Video Diffusion |

---

### Part L: 多模态与对齐 {#part-l}
*Multimodal & Alignment*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 121 | 对比学习 (SimCLR, MoCo) | Contrastive Learning |
| 122 | CLIP | Contrastive Language-Image Pretraining |
| 123 | 多模态融合策略 | Early vs Late Fusion |
| 124 | 指令微调 | Instruction Tuning |
| 125 | 适配器与 LoRA | Adapters & Low-Rank Adaptation |

---

## 第五卷：控制、强化与代理 {#vol5}
### Volume V: Agents, Control & Dynamics

> *不仅要感知，还要行动。引入时间、反馈和奖励。*

---

### Part M: 强化学习基础 {#part-m}
*RL Foundations*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 126 | 马尔可夫决策过程 (MDP) | MDP |
| 127 | 贝尔曼方程与最优性原理 | Bellman Equation |
| 128 | 动态规划 | Value & Policy Iteration |
| 129 | 蒙特卡洛方法与时序差分 | MC & TD Learning |
| 130 | Q-Learning 与 DQN | Deep Q-Networks |

---

### Part N: 策略优化与高级 RL {#part-n}
*Policy Optimization*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 131 | 策略梯度定理 | Policy Gradient Theorem |
| 132 | REINFORCE 算法与基线 | REINFORCE |
| 133 | Actor-Critic 架构 | A2C/A3C |
| 134 | 信任域策略优化 (TRPO) | TRPO |
| 135 | 近端策略优化 (PPO) — ChatGPT 的训练基石 | PPO |
| 136 | 软演员-评论家 (SAC) | SAC & Max-Entropy RL |
| 137 | 离线强化学习与 CQL | Offline RL |
| 138 | 逆强化学习 | Inverse RL |
| 139 | 多智能体强化学习 | MARL |
| 140 | 课程学习 | Curriculum Learning |

---

### Part O: 模型、规划与世界模型 {#part-o}
*Planning & World Models*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 141 | 基于模型的 RL | Model-Based RL |
| 142 | 蒙特卡洛树搜索 (MCTS) — AlphaGo 的核心 | MCTS |
| 143 | 规划即推断 | Planning as Inference |
| 144 | 预测编码 | Predictive Coding |
| 145 | 自由能原理 (Friston) | Free Energy Principle |
| 146 | 联合嵌入预测架构 (JEPA) | JEPA / LeCun's World Model |
| 147 | 潜在空间的动力学模型 | Dreamer / MuZero |
| 148 | 控制理论与 PID 控制器 | Control Theory & PID |
| 149 | 线性二次调节器 (LQR) | LQR |
| 150 | 模型预测控制 (MPC) | MPC |

---

## 第六卷：前沿、理论与新范式 {#vol6}
### Volume VI: Frontiers & New Paradigms

> *现有的问题与未来的解法。这是构建新理论的直接跳板。*

---

### Part P: 学习理论前沿 {#part-p}
*Frontiers of Learning Theory*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 151 | VC 维与 Rademacher 复杂度 | VC Dimension |
| 152 | 偏差-方差权衡与双重下降 | Double Descent |
| 153 | 深度学习的彩票假设 | Lottery Ticket Hypothesis |
| 154 | 泛化误差界 | Generalization Bounds |
| 155 | 缩放定律的数学形式 | Scaling Laws |
| 156 | 涌现能力的相变理论 | Emergent Abilities |
| 157 | 信息瓶颈理论 | Information Bottleneck |
| 158 | 灾难性遗忘与持续学习 | Lifelong Learning |
| 159 | 机器遗忘 | Machine Unlearning |
| 160 | 对抗鲁棒性的几何界 | Adversarial Robustness |

---

### Part Q: 新型架构与计算范式 {#part-q}
*New Paradigms*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 161 | 状态空间模型 (SSM) 与 Mamba | SSM & Mamba |
| 162 | 线性注意力机制 | Linear Attention |
| 163 | 神经算子 (FNO) — PDE 求解 | Neural Operators |
| 164 | **柯普曼算子理论** — 你的核心领域 | **Koopman Operator Theory** |
| 165 | 动力系统深度学习 | Deep Learning for Dynamical Systems |
| 166 | 物理信息神经网络 (PINNs) | PINNs |
| 167 | 脉冲神经网络 (SNN) | Spiking Neural Networks |
| 168 | 储蓄池计算 (ESN) | Reservoir Computing |
| 169 | 超维计算 | Hyperdimensional Computing |
| 170 | 量子机器学习基础 | Quantum ML |

---

### Part R: 逻辑、因果与推理 {#part-r}
*Logic, Causality & Reasoning*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 171 | 符号人工智能与逻辑 | Symbolic AI |
| 172 | 神经符号 AI | Neuro-Symbolic Integration |
| 173 | 因果推断与 Do-Calculus | Causal Inference (Pearl) |
| 174 | 结构因果模型 (SCM) | Structural Causal Models |
| 175 | 发现因果结构 | Causal Discovery |
| 176 | 反事实推理 | Counterfactual Reasoning |
| 177 | 思维链的机制分析 | Chain-of-Thought |
| 178 | 检索增强生成 (RAG) | RAG |
| 179 | 具身智能与 Sim2Real | Embodied AI |
| 180 | 联邦学习与隐私计算 | Federated Learning |

---

### Part S: 你的新理论预备 {#part-s}
*Prelude to The New Theory*

> *解构当前 AI 的"阿喀琉斯之踵"，为新范式做铺垫。*

| 章节 | 标题 | 英文 |
|:----:|------|------|
| 181 | 反向传播的生物不可信性 | Biological Implausibility of Backprop |
| 182 | 全局误差信号的局限性 | Limitations of Global Error Signals |
| 183 | OOD 泛化的根本困境 | The Fundamental OOD Dilemma |
| 184 | 为什么只有预测是不够的？ | Why Prediction is Not Enough |
| 185 | 能量视角的回归 | The Return of Energy Perspectives |
| 186 | 局部学习规则 (Hebbian) | Local Learning Rules |
| 187 | 自组织系统 | Self-Organizing Systems |
| 188 | 复杂系统与混沌边缘 | Edge of Chaos |
| 189 | 智能的热力学代价 | Thermodynamic Cost of Intelligence |
| **190–200** | **[你的理论命名空间]** | **Reserved for Survival-Gated Koopman Dynamics** |

---

<div style="text-align:center; margin: 2em 0;">

[← 返回全书目录](/book/) · [中册：数学原理 →](/book/volume2/)

</div>
