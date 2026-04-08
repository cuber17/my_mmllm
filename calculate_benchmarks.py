import json
import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import wordpunct_tokenize

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate benchmark metrics for generated captions.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/jyz/my_mmLLM",
        help="Project root directory.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help="Path to result json file (contains predicted_caption and texts_ground_truth).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Path to save benchmark metrics json.",
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="Hugging Face endpoint mirror.",
    )
    return parser.parse_args()


def simple_tokenize(text):
    """Resource-free tokenizer: avoids NLTK punkt/punkt_tab dependency."""
    if text is None:
        return []
    return wordpunct_tokenize(str(text).lower())


def fallback_meteor_like(pred_tokens, ref_tokens_list):
    """A lightweight fallback when NLTK wordnet resource is unavailable.

    Uses best unigram F1 against multiple references.
    """
    pred_set = set(pred_tokens)
    if not pred_set:
        return 0.0

    best_f1 = 0.0
    for ref_tokens in ref_tokens_list:
        ref_set = set(ref_tokens)
        if not ref_set:
            continue
        overlap = len(pred_set & ref_set)
        if overlap == 0:
            continue
        precision = overlap / len(pred_set)
        recall = overlap / len(ref_set)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

class BenchmarkEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f">>> Initializing Extractors on {self.device}...")
        
        # 1. SBERT Model (通用语义相似度)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # 2. SimCSE Model (对比学习优化的句向量，擅长判断相似性)
        # 这里使用 princeton-nlp 的 SimCSE 或者兼容的 SBERT 模型
        self.simcse_model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased', device=self.device)
        
        # 3. ROUGE Scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Smooth function for BLEU
        self.smooth = SmoothingFunction().method1

    def compute_sbert_sim(self, preds, refs):
        """计算 SBERT 余弦相似度"""
        embeddings1 = self.sbert_model.encode(preds, convert_to_tensor=True)
        embeddings2 = self.sbert_model.encode(refs, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # 取对角线元素 (一一对应)
        return torch.diag(cosine_scores).mean().item()

    def compute_simcse_sim(self, preds, refs):
        """计算 SimCSE 余弦相似度"""
        embeddings1 = self.simcse_model.encode(preds, convert_to_tensor=True)
        embeddings2 = self.simcse_model.encode(refs, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return torch.diag(cosine_scores).mean().item()

    def compute_bertscore(self, preds, refs):
        """计算 BERTScore (使用 F1 作为最终分数)"""
        if bert_score is None:
            raise ImportError(
                "bert-score is not installed. Please run: pip install bert-score"
            )

        _, _, f1 = bert_score(
            preds,
            refs,
            lang='en',
            device=self.device,
            rescale_with_baseline=True
        )
        return torch.as_tensor(f1).float().mean().item()

    def compute_traditional_metrics(self, preds, refs_list):
        """
        计算 BLEU, ROUGE, METEOR
        注意: refs_list 是 list of lists, 因为一个样本可能有多个 GT
        """
        bleu1_scores = []
        bleu4_scores = []
        rouge_l_scores = []
        meteor_scores = []

        for pred, gt_candidates in zip(preds, refs_list):
            # Tokenize
            pred_tokens = simple_tokenize(pred)
            gt_tokens_list = [simple_tokenize(g) for g in gt_candidates]

            # --- BLEU ---
            bleu1 = sentence_bleu(gt_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smooth)
            bleu4 = sentence_bleu(gt_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth)
            bleu1_scores.append(bleu1)
            bleu4_scores.append(bleu4)

            # --- ROUGE-L ---
            # Rouge scorer handles multiple references by taking the max, mostly expects strings
            # 这里简单处理，取所有GT中最高的 ROUGE 分数
            pred_text = str(pred)
            r_scores = [self.rouge_scorer.score(str(g), pred_text)['rougeL'].fmeasure for g in gt_candidates]
            rouge_l_scores.append(max(r_scores) if r_scores else 0.0)

            # --- METEOR ---
            # NLTK meteor expects list of strings for reference provided tokenized
            # 注意: 新版 nltk meteor_score 接收的是 list of tokens 和 tokens
            try:
                # 尝试适配不同版本的 NLTK
                m_score = meteor_score(gt_tokens_list, pred_tokens)
            except (AttributeError, LookupError):
                 # 旧版可能传字符串
                 try:
                     m_score = meteor_score(gt_candidates, pred)
                 except Exception:
                     # 如果缺少 wordnet 等资源，则使用离线近似分数兜底
                     m_score = fallback_meteor_like(pred_tokens, gt_tokens_list)
            meteor_scores.append(m_score)

        return {
            "BLEU-1": np.mean(bleu1_scores),
            "BLEU-4": np.mean(bleu4_scores),
            "ROUGE-L": np.mean(rouge_l_scores),
            "METEOR": np.mean(meteor_scores)
        }

def build_default_output_path(input_file):
    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    if input_name.startswith("test_result_"):
        suffix = input_name[len("test_result_"):]
    else:
        suffix = input_name
    return os.path.join(input_dir, f"benchmark_metrics_{suffix}")


def main():
    args = parse_args()

    base_dir = args.base_dir
    input_file = args.input_file if args.input_file else f"{base_dir}/processed_dataset/test_result_phi4mini_epoch_0_with_98.json"
    output_file = args.output_file if args.output_file else build_default_output_path(input_file)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    print(f"Using HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    print(f"Loading results from {input_file}...")
    data = load_data(input_file)
    
    # 提取 Prediction 和 Ground Truth
    # text_ground_truth 是 list of strings
    # predicted_caption 是 string
    
    valid_data = [d for d in data if 'predicted_caption' in d and 'texts_ground_truth' in d]
    
    preds = [d['predicted_caption'] for d in valid_data]
    # GT list (nested list for traditional metrics)
    refs_list = [d['texts_ground_truth'] for d in valid_data]
    # For embedding metrics, we usually compare against the first GT or average pairwise.
    # Here we take the first GT for embedding speed comparison
    refs_single = [d['texts_ground_truth'][0] for d in valid_data]

    print(f"evaluating {len(preds)} samples...")

    evaluator = BenchmarkEvaluator()

    # 1. 计算 Traditional Metrics
    print("Calculating BLEU, ROUGE, METEOR...")
    trad_metrics = evaluator.compute_traditional_metrics(preds, refs_list)

    # 2. 计算 Semantic Metrics
    print("Calculating SBERT Similarity...")
    sbert_score = evaluator.compute_sbert_sim(preds, refs_single)
    
    print("Calculating SimCSE Similarity...")
    simcse_score = evaluator.compute_simcse_sim(preds, refs_single)

    print("Calculating BERTScore...")
    bert_score_f1 = evaluator.compute_bertscore(preds, refs_single)

    # 3. 汇总结果
    results = {
        **trad_metrics,
        "SBERT": sbert_score,
        "SimCSE": simcse_score,
        "BERTScore": bert_score_f1
    }

    print("\n" + "="*40)
    print(" FINAL BENCHMARK RESULTS")
    print("="*40)
    for k, v in results.items():
        print(f"{k:<15}: {v:.4f}")
    print("="*40)

    # 保存
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"metrics saved to {output_file}")

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"Using HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    main()