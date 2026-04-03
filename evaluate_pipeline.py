import json
import os
import argparse
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from inference_demo import MMExpertInference


LLM_PRESETS = {
    "phi3mini": "Phi-3-mini-4k-instruct",
    "phi35": "Phi-3.5-mini-instruct",
    "qwen25": "Qwen2.5-3B-Instruct",
}

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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mmLLM pipeline with selectable LLM backend")
    parser.add_argument("--base_dir", type=str, default="/root/jyz/my_mmLLM")
    parser.add_argument("--llm_key", type=str, default="phi3mini", choices=list(LLM_PRESETS.keys()))
    parser.add_argument("--llm_base_path", type=str, default="", help="Override base LLM path")
    parser.add_argument("--lora_path", type=str, default="", help="Stage2 epoch directory that contains LoRA adapter")
    parser.add_argument("--projector_path", type=str, default="", help="Override projector checkpoint path")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--test_json", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--attr_exp_id", type=str, default="attributes_20260131_145653")
    parser.add_argument("--radar_ckpt", type=str, default="")
    parser.add_argument("--confidence_threshold", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. 路径和环境配置 ---
    BASE_DIR = args.base_dir
    DATA_ROOT = args.data_root if args.data_root else f"{BASE_DIR}/processed_dataset/"
    TEST_JSON = args.test_json if args.test_json else f"{DATA_ROOT}/test.json"
    default_out = f"{DATA_ROOT}/test_result_{args.llm_key}.json"
    OUTPUT_JSON = args.output_json if args.output_json else default_out

    # 模型 Checkpoint 路径 (请确保与 logs 文件夹中实际存在的一致)
    # 属性模型
    ATTR_EXP_ID = args.attr_exp_id
    ATTR_CKPT = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/best.pth"
    ATTR_MAP = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/label_maps.json"
    
    # Stage 1 Encoder
    RADAR_CKPT = args.radar_ckpt if args.radar_ckpt else f"{BASE_DIR}/logs/clip_20260120_224659/radar_encoder_only.pth"
    
    # Stage 2 Projector & LoRA: 默认需要你通过 --lora_path 指定当前实验目录
    if args.lora_path:
        LORA_PATH = args.lora_path
    else:
        raise ValueError("Please provide --lora_path to the trained Stage2 epoch directory.")

    PROJ_CKPT = args.projector_path if args.projector_path else os.path.join(LORA_PATH, "projector.pth")
    
    # LLM Base
    if args.llm_base_path:
        LLM_BASE = args.llm_base_path
    else:
        llm_dir = LLM_PRESETS[args.llm_key]
        LLM_BASE = os.path.join(BASE_DIR, "huggingface", llm_dir)

    if not os.path.exists(LLM_BASE):
        raise FileNotFoundError(f"LLM base path not found: {LLM_BASE}")

    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA path not found: {LORA_PATH}")

    if not os.path.exists(PROJ_CKPT):
        raise FileNotFoundError(f"Projector checkpoint not found: {PROJ_CKPT}")

    eval_device = args.device
    if eval_device == "cuda" and not torch.cuda.is_available():
        eval_device = "cpu"
        print("CUDA is not available, fallback to CPU.")

    print("=== Evaluation Config ===")
    print(f"LLM Key      : {args.llm_key}")
    print(f"LLM Base     : {LLM_BASE}")
    print(f"LoRA Path    : {LORA_PATH}")
    print(f"Projector    : {PROJ_CKPT}")
    print(f"Output JSON  : {OUTPUT_JSON}")
    print(f"Device       : {eval_device}")
    print("=========================")

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
            data_root_dir=DATA_ROOT,   # 传入数据根目录
            device=eval_device,
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # --- 3. 加载测试数据列表 ---
    with open(TEST_JSON, 'r') as f:
        data_list = json.load(f)
    
    # [新增] 定义置信度阈值
    CONFIDENCE_THRESHOLD = args.confidence_threshold
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