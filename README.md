# mmLLM: 基于毫米波雷达感知的多模态大语言模型

本项目提出了一种端到端的“毫米波信号 → 结构化语义 → LLM 推理 → 自然语言描述”的框架。系统从毫米波雷达的 TD/TR/TA 热图出发，通过“语义编码 + 属性解耦感知”两条并行路径，实现对人体活动的深度理解与自然语言生成。

## 方法论与系统架构

本系统采用两阶段、三模块的设计方案：

### 1. 多视图雷达输入 (Multi-View Radar Input)
系统输入数据的基石是毫米波雷达生成的三个正交视图热图，它们从不同维度刻画了人体运动：
*   **TD (Range-Doppler)**: 距离-多普勒图，反映目标的运动速度与距离关系。
*   **TR (Range-Time)**: 距离-时间图，记录距离随时间的动态变化轨迹。
*   **TA (Range-Angle)**: 距离-角度图，反映目标的空间方位信息。

### 2. 双流感知主干 (Dual-Stream Perception Backbone)
为了兼顾“可解释性”与“特征丰富性”，我们设计了两条并行的感知路径：

*   **路径一：隐式语义编码 (Implicit Semantic Stream)**
    *   **模型**: Vision Transformer (ViT-Base)
    *   **机制**: 将拼接后的 TD/TR/TA 热图视为视频帧序列或多通道图像。利用 Self-Attention 机制捕获长距离的时空依赖和微多普勒模式。
    *   **输出**: 连续的高维嵌入向量 (Embedding)。这部分特征类似于人类的“直觉”，包含了难以用离散标签描述的运动韵律和细微动态。

*   **路径二：显式属性解耦 (Explicit Attribute Stream)**
    *   **模型**: Multi-Head ResNet-18
    *   **机制**: 一个轻量级的卷积神经网络，通过多任务学习头（Multi-Head Classifiers）并行解耦出物理属性。
    *   **输出**: 结构化的文本标签 (Action, Posture, Intensity, Body Part)。
    *   **作用**: 提供硬性的物理约束 (Physics-aware Constraints)，作为 Prompt 显式输入给 LLM，有效防止大模型的“幻觉”问题。

### 3. 多模态融合推理 (LLM Reasoning)
*   **基座模型**: [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (3.8B)
*   **对齐层 (Projector)**: 使用 MLP 将 ViT 的雷达特征映射到 LLM 的词向量空间。
*   **生成过程**: LLM 接收 `[Radar Tokens] + [Attribute Text Prompt] + [User Instruction]`，结合感知的隐式特征与显式事实，生成流畅、准确的自然语言描述。


## 项目结构
```bash
my_mmLLM/
├── requirements.txt            # 项目依赖
├── README.md                   # 项目说明文档
├── inference_demo.py           # 单样本/交互式推理脚本
├── visualize_radar.py          # 雷达热图可视化工具
├── evaluate_pipeline.py        # 批量评测脚本 (生成结果 + 计算指标)
│
├── src/                        # 核心源码
│   ├── clip/                   # Stage 1: 对比学习 (ViT Encoder)
│   ├── attributes_perception/  # Stage 1: 属性分类 (ResNet Attributes)
│   └── llm/                    # Stage 2: 投影层与数据加载
│
├── logs/                       # 训练日志与模型权重保存目录
│   ├── clip_2026.../           # 存放训练好的 Radar Encoder
│   ├── attributes_2026.../     # 存放训练好的 Attribute Model
│   └── stage2_2026.../         # 存放训练好的 Projector 和 LoRA 权重
│
├── processed_dataset/          # 数据集目录
│   ├── train.json / test.json  # 数据索引
│   ├── imgs/                   # 训练集 .npy 文件
│   └── imgs_test/              # 测试集 .npy 文件
│
└── huggingface/                # 预下载的 LLM 模型权重
```

## 快速开始
### 1. 环境配置 (Installation)

```bash
# 创建环境
conda create -n mmLLM python=3.10 -y
conda activate mmLLM

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch torchvision torchaudio

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备 (Data Preparation)
请确保您的数据集已解压至 `processed_dataset/` 目录，需包含：
*   **Visual Data**: `.npy` 格式的 TD/TR/TA 热图文件。
*   **Metadata**: `train.json` 和 `test.json`，格式如下：
    ```json
    [
      {
        "id": "000001",
        "td_path": "./imgs/000001_td.npy",
        "tr_path": "./imgs/000001_tr.npy",
        "ta_path": "./imgs/000001_ta.npy",
        "labels": {"action": "walk", "posture": "standing", ...},
        "texts_ground_truth": ["A person is walking forward."]
      }
    ]
    ```

### 3. 数据可视化工具 (Visualization)
在开始训练前，可以使用提供的脚本检查数据质量：
```bash
# 交互式查看
python visualize_radar.py

# 查看指定文件
python visualize_radar.py ./processed_dataset/imgs/011761_td.npy
```
生成的图像将保存在 `vis_outputs/` 目录下。

## 训练策略 (Training Strategy)

训练过程分为两个阶段进行。
### Stage 1: 单模态预训练 (Unimodal Pre-training)
目标：让编码器学会“看”雷达图，让属性网络学会“识别”动作。

**步骤 1.1: 训练 CLIP 对齐 (Radar Encoder)**
使用对比学习对齐雷达特征与文本特征。
```bash
python train_clip.py
# 输出权重: logs/clip_TIMESTAMP/radar_encoder_only.pth
```

**步骤 1.2: 训练属性分类器 (Attribute Classifier)**
训练 ResNet 对动作属性进行多标签分类。
```bash
python train_attr.py
# 输出权重: logs/attributes_TIMESTAMP/best.pth
```

### Stage 2: 多模态指令微调 (Multimodal Instruction Tuning)
目标：冻结感知模块，训练 Projector 和 LLM 适配器，使其学会根据雷达信息“说话”。

**步骤 2.1: 配置路径**
修改 train_stage2.py 中的 `RADAR_ENCODER_PATH` 指向 Stage 1 训练好的权重。

**步骤 2.2: 启动微调**
```bash
python train_stage2.py
# 输出权重: logs/stage2_TIMESTAMP/epoch_X/projector.pth & adapter_model.bin
```

## 推理与评测 (Inference)

### 1. 交互式推理 (Inference Demo)
加载训练好的所有组件，通过命令行输入样本 ID 进行测试：
```bash
python inference_demo.py
```
> **注意**: 请先在 inference_demo.py 的 `if __name__ == "__main__":` 部分修改模型路径为你实际训练好的 Checkpoint 路径。

### 2. 批量评测 (Benchmark Evaluation)
在测试集上生成所有结果并计算 BLEU, ROUGE, METEOR, SimCSE 等核心指标：
```bash
python evaluate_pipeline.py
```

## 实验结果 (Results)

### 1. 属性感知准确率
Explicit Path (ResNet) 为 LLM 提供了关键的事实依据，其各项属性的 Top-1 准确率如下：

| Attributes | Accuracy |
| :--- | :---: |
| Action Category | **78.70%** |
| Posture | **94.70%** |
| Intensity | **83.20%** |
| Active Part | **81.20%** |
| Trajectory | **85.50%** |
| **Average** | **84.66%** |

### 2. 生成文本质量
得益于双流架构，MMExpert 在语义一致性指标上显著优于传统 Baseline：

| Metric | Score | 说明 |
| :--- | :---: | :--- |
| **BLEU-1** | 0.537 | 词汇级准确度 |
| **BLEU-4** | 0.145 | 语句流畅性 |
| **ROUGE-L** | 0.439 | 信息召回率 |
| **METEOR** | 0.450 | 语义对齐质量 |
| **SBERT-Sim** | 0.566 | 句子嵌入相似度 |
| **SimCSE-Sim** | **0.703** | 语义对比相似度 (High!) |