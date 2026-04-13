import torch
import torch.nn.functional as F  # [新增] 需要用到 Softmax
import json
import os
from torchvision import transforms
from PIL import Image
import numpy as np

# 尝试兼容不同的导入方式（直接运行 vs 被其他模块调用）
try:
    from model import MultiHeadAttributeClassifier
except ImportError:
    from .model import MultiHeadAttributeClassifier

class AttributePredictor:
    def __init__(self, checkpoint_path, label_maps_path, device='cuda', temperature=5.5):
        self.device = torch.device(device)
        # Temperature scaling: T > 1 flattens probabilities and reduces over-confidence.
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(label_maps_path):
            raise FileNotFoundError(f"Label map not found: {label_maps_path}")

        # 加载 Label Map
        with open(label_maps_path, 'r') as f:
            self.label_maps = json.load(f)
            
        # 反转 Map (id -> str) 用于解码
        self.id2label = {}
        for attr, mapping in self.label_maps.items():
            self.id2label[attr] = {v: k for k, v in mapping.items()}
        
        output_dims = {k: len(v) for k, v in self.label_maps.items()}
        
        # 加载模型
        self.model = MultiHeadAttributeClassifier(output_dims).to(self.device)
        print(f"Loading model from {checkpoint_path}...")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        # 预处理必须与训练保持一致：Resize -> Tensor -> Normalize
        # 注意：这里假设输入已经是 Tensor 且经过了 dataset 里的 MinMax 和 Resize
        # 如果是原始 numpy 输入，需要外部先处理或是扩展这个 transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict_attributes(self, img_tensor, threshold=0.6): # [修改] 新增 threshold 参数
        """
        输入: Tensor (3, H, W) 或 (1, 3, H, W)
        输出: 结构化字典 + 文本 Prompt
        Trick: 只有当某属性的预测置信度 > threshold 时，才将其加入 prompt
        置信度计算使用 temperature-scaled softmax:
            p = softmax(logits / T), T > 1 时分布更均匀
        """
        # 确保是 Tensor
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor).float()

        # 增加 Batch 维度
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        img_tensor = self.transform(img_tensor).to(self.device)
        
        with torch.no_grad():
            preds = self.model(img_tensor)
        
        results = {}
        prompt_parts = []
        
        # 定义模板
        templates = {
            'action_category': "Action category is {}",
            'posture': "Posture is {}",
            'intensity': "Intensity is {}",
            'active_part': "Main active part is {}",
            'trajectory': "Motion trajectory is {}"
        }
        
        # 按照特定顺序生成描述
        order = ['action_category', 'posture', 'trajectory', 'active_part', 'intensity']
        
        for k in order:
            if k in preds:
                logits = preds[k] # (Batch, Num_Classes)
                # Temperature scaling: larger T -> flatter probability distribution.
                scaled_logits = logits / self.temperature
                probs = F.softmax(scaled_logits, dim=1)
                max_prob, idx = torch.max(probs, dim=1)
                
                confidence = max_prob.item()
                idx = idx.item()
                
                # [新增] 阈值判断逻辑
                # 如果置信度高于阈值，才予以采纳
                if confidence >= threshold:
                    label_str = self.id2label[k][idx]
                    results[k] = {
                        'label': label_str,
                        'confidence': confidence
                    }
                    prompt_parts.append(templates[k].format(label_str))
                else:
                    # 也可以选择记录为 "Unknown" 或者完全忽略
                    results[k] = {
                        'label': 'uncertain',
                        'confidence': confidence
                    }
            
        # 拼接 Prompt
        if len(prompt_parts) > 0:
            full_text_prompt = ". ".join(prompt_parts) + "."
        else:
            # 如果依然所有属性都低于阈值，返回一个通用的空提示，避免空字符串
            full_text_prompt = "" 
        
        return results, full_text_prompt

if __name__ == "__main__":
    # 使用示例：测试刚刚训练好的模型
    import sys
    
    # 模拟一个随机输入 (3通道热图)
    dummy_input = torch.randn(3, 224, 224)
    
    # 路径配置
    EXP_ID = "experiment_20260118_185816" # 你的最新实验ID
    BASE_DIR = os.path.join("logs", EXP_ID)
    ckpt = os.path.join(BASE_DIR, "best.pth")
    maps = os.path.join(BASE_DIR, "label_maps.json")
    
    if os.path.exists(ckpt):
        predictor = AttributePredictor(ckpt, maps)
        res, prompt = predictor.predict_attributes(dummy_input)
        
        print("\n--- Inference Result ---")
        print(f"Structured: {res}")
        print(f"LLM Prompt: {prompt}")
    else:
        print(f"Checkpoint not found at {ckpt}, please check path")