# mmLLM: 基于毫米波雷达感知的多模态大语言模型

本项目提出了一种端到端的“毫米波信号 → 结构化语义 → LLM 推理 → 自然语言描述”的框架。系统从毫米波雷达的 TD/TR/TA 热图出发，通过“语义编码 + 属性解耦感知”两条并行路径，实现对人体活动的深度理解与自然语言生成。

## 项目简介

本系统采用 **两阶段、两条主干** 的架构设计，旨在解决雷达信号语义与自然语言空间对齐难的问题：

1.  **Stage 1: 语义提取 (Semantic Representation)**
    *   **路径一：毫米波语义编码 (Encoder Path)**
        *   输入：TD / TR / TA 热图
        *   输出：连续的语义嵌入向量 (Embedding)
        *   作用：捕获运动节奏、微多普勒形态等难以用离散标签描述的“隐式语义”。
    *   **路径二：动作属性解耦 (Attribute Disentanglement Path)**
        *   输入：TD / TR / TA 热图
        *   模型：多头 CNN / 多任务分类网络
        *   输出：结构化的动作属性标签 (Action, Posture, Intensity, Part)
        *   作用：提供可解释、具有物理约束的“显式事实信息”，作为 Prompt 辅助 LLM。
        *   *Trick*: 引入置信度阈值机制，只保留高置信度预测结果，兼顾准确性与泛化性。

2.  **Stage 2: LLM 融合理解 (LLM Reasoning)**
    *   基座模型：[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
    *   输入：Path 1 的 Embedding + Path 2 的属性 Prompt文本
    *   作用：融合连续语义与离散属性，生成流畅、准确的人体活动描述。

## 项目结构
```bash
├── README.md
├── report.md
├── requirements.txt
├── huggingface/               
├── inference_demo.py
├── logs/
├── processed_dataset/
├── src
│   ├── __pycache__
│   ├── attributes_perception/
│   ├── clip/
│   ├── llm/
├── calculate_benchmarks.py
├── download_model.py
├── evaluate_pipeline.py
├── test_torch.py
├── train_attr.py
├── train_clip.py
└── train_stage2.py
```

## 环境配置

本项目基于 PyTorch 开发，请按照以下步骤搭建环境：

- 创建环境并安装依赖
```bash
conda create -n mmLLM python=3.10 -y
conda activate mmLLM
pip install -r requirements.txt
```

## 快速开始
1. 数据准备
确保数据集位于 processed_dataset 目录下，包含：

test.json: 测试集索引文件
相关雷达热图数据
2. 推理生成 (Inference)
运行推理脚本，加载训练好的 Checkpoints 并生成描述：

注意: 请在 evaluate_pipeline.py 中确认 ATTR_CKPT (属性模型), RADAR_CKPT (Encoder), PROJ_CKPT (Projector) 的路径是否正确指向 logs 下的文件。

3. 指标计算 (Benchmark)
生成结果后，计算 BLEU, ROUGE, METEOR, SBERT-Sim, SimCSE-Sim 等指标：

## 实验结果
1. 属性解耦模块性能
属性解耦模块负责提供显式的物理约束，其在测试集上的 Top-1 准确率如下：

属性类别	Accuracy
Action Category	78.70%
Posture	94.70%
Intensity	83.20%
Active Part	81.20%
Trajectory	85.50%
Average	84.66%
2. 生成文本质量评测 (Benchmark)
本模型 (mmExpert) 在语义一致性与文本质量上均表现优异，具体指标如下：

Metric	Score	说明
BLEU-1	0.537	词汇级精准度
BLEU-4	0.145	4-gram 连贯性
ROUGE-L	0.439	最长公共子序列 recall
METEOR	0.450	综合语义匹配
SBERT-Sim	0.566	句子级语义相似度
SimCSE-Sim	0.703	对比学习语义相似度
与 Baseline (RadarLLM 等) 相比，本方法在 BLEU 与 SimCSE 上有显著提升，证明引入属性解耦路径有效增强了生成描述的准确性和可解释性。