## 试验记录
### 属性解耦头

- 训练集：`train.json`，共11364个样本, 测试集：`test.json`，共1000个样本
```bash
==============================
Evaluation Results (Top-1 Accuracy)
==============================
action_category     : 80.50%
posture             : 95.40%
intensity           : 82.80%
active_part         : 80.60%
trajectory          : 84.10%
------------------------------
Average Accuracy    : 84.68%
==============================
```

```bash
Experiment: experiment_20260118_185816
Evaluation Results (Top-1 Accuracy)
========================================
action_category     : 78.70%
posture             : 94.70%
intensity           : 83.20%
active_part         : 81.20%
trajectory          : 85.50%
----------------------------------------
Average Accuracy    : 84.66%
```

### benchmark指标
- RadarLLM 结果
```json
{
    "ROUGE-1": 0.384,
    "ROUGE-L": 0.360,
    "BLEU-1": 0.480,
    "BLEU-4": 0.114,
    "METEOR": 0.337,
    "CIDEr": 8.3,
    "BERTScore": 0.833,
    "SimCSE": 0.896
}

- mmExpert WaveLLM-CLIP-ViT 结果
{
    "BLEU-1": 0.469,
    "ROUGE-L": 0.499,
    "METEOR": 0.235,
    "SBERT-Sim": 0.722,
    "SimCSE-Sim": 0.712,
}


- 模型配置与 Checkpoint 路径
```python
# 模型 Checkpoint 路径 (请确保与 logs 文件夹中实际存在的一致)
# 属性模型
ATTR_EXP_ID = "experiment_20260118_185816"
ATTR_CKPT = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/best.pth"
ATTR_MAP = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/label_maps.json"

# Stage 1 Encoder
RADAR_CKPT = f"{BASE_DIR}/logs/clip_20260120_224659/radar_encoder_only.pth"

# Stage 2 Projector & LoRA
# 注意: 如果 logs 里没有 epoch_2, 请改为 epoch_1 或 epoch_0
STAGE2_EPOCH = "epoch_2" 
STAGE2_DIR = f"{BASE_DIR}/logs/stage2_20260121_210018/{STAGE2_EPOCH}"
PROJ_CKPT = f"{STAGE2_DIR}/projector.pth"
LORA_PATH = f"{STAGE2_DIR}"
```
- epoch9 阈值设为 0.9 测评结果
```json
{
    "BLEU-1": 0.5431045762459199,
    "BLEU-4": 0.1896712487648248,
    "ROUGE-L": 0.46229752658324264,
    "METEOR": 0.47393958903066086,
    "SBERT-Sim": 0.5900965929031372,
    "SimCSE-Sim": 0.7091032862663269
}
```

## Qwen2.5-3B

- 批量评测命令
```bash
python evaluate_pipeline.py   --llm_key qwen25   --llm_base_path /root/jyz/my_mmLLM/huggingface/Qwen2.5-3B-Instruct   --lora_path logs/stage2_qwen25_3b_20260402_115702/epoch_9   --output_json /root/jyz/my_mmLLM/processed_dataset/test_result_qwen25_epoch_9_with_90.json   --confidence_threshold 0.9
```

### benchmark指标

## Phi-3.5-mini
- 批量评测命令
```bash
python evaluate_pipeline.py --llm_key phi35  --llm_base_path /root/jyz/my_mmLLM/huggingface/Phi-3.5-mini-instruct  --lora_path logs/stage2_phi35_20260403_031610/epoch_9 --output_json /root/jyz/my_mmLLM/processed_dataset/test_result_phi35_epoch_9_with_90.json  --confidence_threshold 0.9
```

## Phi-4-mini
- 批量评测命令
```bash
python evaluate_pipeline.py --llm_key phi4mini --lora_path /root/jyz/my_mmLLM/logs/stage2_phi4mini_20260404_065504/epoch_6 --output_json /root/jyz/my_mmLLM/processed_dataset/test_result_phi4mini_epoch_6_with_90.json --confidence_threshold 0.9
```

## Gemma-2-1.5B
- 批量评测命令
```bash
python evaluate_pipeline.py --llm_key gemma2_2b --lora_path /root/jyz/my_mmLLM/logs/stage2_gemma2_2b_20260404_074536/epoch_9 --output_json /root/jyz/my_mmLLM/processed_dataset/test_result_gemma2_2b_epoch_9_with_90.json --confidence_threshold 0.9
```