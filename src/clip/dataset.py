import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class MMClipDataset(Dataset):
    def __init__(self, json_file, root_dir, tokenizer=None, max_text_len=77, transform=None):
        """
        Args:
            json_file: 包含路径和caption的json文件
            root_dir: 图片根目录
            tokenizer: HuggingFace Tokenizer (如果需要在Dataset里tokenization)
            max_text_len: 文本最大长度
        """
        self.root_dir = root_dir
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.transform = transform
        
        # 目标尺寸，CLIP常用的尺寸
        self.target_size = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # -----------------------------
        # 1. 处理雷达图像 (核心：解决尺寸不统一)
        # -----------------------------
        td, tr, ta = self._load_images(item)
        
        # 定义处理函数: 转 Tensor -> (1,H,W) -> Resize
        def process_img(arr):
            arr = arr.astype(np.float32)
            # Min-Max 归一化 [0, 1]
            if arr.max() - arr.min() > 1e-6:
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            
            t = torch.from_numpy(arr)
            # 补齐维度
            if t.ndim == 2: 
                t = t.unsqueeze(0) # (H, W) -> (1, H, W)
            elif t.ndim == 3 and t.shape[2] == 1: 
                t = t.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
                
            # 强制 Resize 到 224x224
            # align_corners=False 避免虽然尺寸对了但特征偏移
            t = F.interpolate(t.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            return t

        try:
            td_t = process_img(td)
            tr_t = process_img(tr)
            ta_t = process_img(ta)
            
            # 堆叠成 3 通道 (符合 ViT 输入要求)
            # Shape: (3, 224, 224)
            wave_tensor = torch.cat([td_t, tr_t, ta_t], dim=0)

            # 归一化 (Normalize) 到 [-1, 1] 或 ImageNet 均值
            if self.transform:
                wave_tensor = self.transform(wave_tensor)
                
        except Exception as e:
            print(f"Error loading {item.get('id')}: {e}")
            wave_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        # -----------------------------
        # 2. 处理文本 (Texts)
        # -----------------------------
        # 假设 JSON 里有个 "texts_ground_truth" 列表，我们随机选一个作为正样本
        captions = item.get('texts_ground_truth', [""])
        if isinstance(captions, list) and len(captions) > 0:
            caption = np.random.choice(captions) # 训练时随机选一个增强鲁棒性
        else:
            caption = "unknown mmwave signal"

        # 如果提供了 tokenizer，直接返回处理好的 token
        tokenized_text = None
        if self.tokenizer:
            tokenized_text = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            input_ids = tokenized_text['input_ids'].squeeze(0)
            attention_mask = tokenized_text['attention_mask'].squeeze(0)
        else:
            # 如果没有 tokenizer，返回原始文本，在 collate_fn 或 model 内部处理
            input_ids = torch.tensor([]) 
            attention_mask = torch.tensor([])

        return {
            'pixel_values': wave_tensor,   # 雷达图像 (3, 224, 224)
            'input_ids': input_ids,        # 文本 token ID
            'attention_mask': attention_mask, # 文本 mask
            'raw_caption': caption         # 原始文本（用于调试或评估）
        }

    def _load_images(self, item):
        """ 安全加载逻辑，与你之前的代码一致 """
        try:
            def safe_load(path_key):
                rel_path = item.get(path_key, '')
                if not rel_path: raise ValueError("Path empty")
                
                fname = os.path.basename(rel_path)
                folder = 'imgs_test' if 'imgs_test' in rel_path else 'imgs'
                full_path = os.path.join(self.root_dir, folder, fname)
                return np.load(full_path)

            td = safe_load('td_path')
            tr = safe_load('tr_path')
            ta = safe_load('ta_path')
            return td, tr, ta
        except Exception:
            z = np.zeros((64, 64), dtype=np.float32)
            return z, z, z