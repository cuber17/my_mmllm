import json
import os
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from inference_demo import MMExpertInference

def calculate_metrics(predictions, ground_truths):
    """计算 BLEU-1, BLEU-4"""
    scores = {'bleu1': 0.0, 'bleu4': 0.0}
    smooth = SmoothingFunction().method1
    
    n = len(predictions)
    if n == 0: return scores
    
    for pred, refs in zip(predictions, ground_truths):
        # 简单的分词
        pred_tokens = pred.lower().replace('.', '').split()
        ref_tokens = [r.lower().replace('.', '').split() for r in refs]
        
        scores['bleu1'] += sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        scores['bleu4'] += sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        
    return {k: v / n for k, v in scores.items()}

def main():
    # --- 1. 路径和环境配置 ---
    BASE_DIR = "/root/jyz/my_mmLLM"
    DATA_ROOT = f"{BASE_DIR}/processed_dataset/"
    TEST_JSON = f"{DATA_ROOT}/test.json"
    OUTPUT_JSON = f"{DATA_ROOT}/test_result_epoch_9_with_50.json"

    # 模型 Checkpoint 路径 (请确保与 logs 文件夹中实际存在的一致)
    # 属性模型
    ATTR_EXP_ID = "attributes_20260131_145653"
    ATTR_CKPT = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/best.pth"
    ATTR_MAP = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/label_maps.json"
    
    # Stage 1 Encoder
    RADAR_CKPT = f"{BASE_DIR}/logs/clip_20260120_224659/radar_encoder_only.pth"
    
    # Stage 2 Projector & LoRA
    # 注意: 如果 logs 里没有 epoch_2, 请改为 epoch_1 或 epoch_0
    STAGE2_EPOCH = "epoch_9" 
    STAGE2_DIR = f"{BASE_DIR}/logs/stage2_20260131_122203/{STAGE2_EPOCH}"
    PROJ_CKPT = f"{STAGE2_DIR}/projector.pth"
    LORA_PATH = f"{STAGE2_DIR}"
    
    # LLM Base
    LLM_BASE = f"{BASE_DIR}/huggingface/Phi-3-mini-4k-instruct"

    # --- 2. 初始化模型 ---
    print(">>> Initializing Inference Model...")
    try:
        expert = MMExpertInference(
            radar_encoder_path=RADAR_CKPT,
            projector_path=PROJ_CKPT,
            attr_model_path=ATTR_CKPT,
            attr_label_map=ATTR_MAP,
            llm_base_path=LLM_BASE,
            llm_adapter_path=LORA_PATH,
            data_json_path=TEST_JSON,  # 传入数据索引路径
            data_root_dir=DATA_ROOT    # 传入数据根目录
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # --- 3. 加载测试数据列表 ---
    with open(TEST_JSON, 'r') as f:
        data_list = json.load(f)
    
    # [新增] 定义置信度阈值
    CONFIDENCE_THRESHOLD = 0.50 
    print(f"Starting evaluation with Attribute Confidence Threshold = {CONFIDENCE_THRESHOLD}")
    
    final_results = []
    
    # 用于计算 Metrics
    all_preds = []
    all_gts = []

    # --- 4. 批量推理 ---
    for item in tqdm(data_list, desc="Evaluating"):
        sample_id = item['id']
        
        # 复制原数据，避免修改读取的缓存
        result_item = item.copy()
        
        try:
            # [修改] 调用 generate 时传入阈值
            response, attr_prompt, _ = expert.generate(sample_id, attr_threshold=CONFIDENCE_THRESHOLD)
            
            # --- 新增字段 ---
            result_item['predicted_label'] = attr_prompt
            result_item['predicted_caption'] = response
            
            # 收集用于计算指标
            all_preds.append(response)
            all_gts.append(item['texts_ground_truth']) 
            
        except Exception as e:
            print(f"Error inferencing {sample_id}: {e}")
            result_item['predicted_label'] = "Error"
            result_item['predicted_caption'] = "Error generating caption."

        final_results.append(result_item)

    # --- 5. 计算总体指标 ---
    if len(all_preds) > 0:
        metrics = calculate_metrics(all_preds, all_gts)
        print("\n" + "="*40)
        print("Evaluation Results")
        print("="*40)
        print(f"BLEU-1: {metrics['bleu1']:.4f}")
        print(f"BLEU-4: {metrics['bleu4']:.4f}")
        print("="*40)

    # --- 6. 保存详细结果 ---
    print(f"Saving {len(final_results)} results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"Detailed results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()