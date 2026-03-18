import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class MMLLMDataset(Dataset):
    def __init__(self, json_file, root_dir):
        self.root_dir = root_dir
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.target_size = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 加载雷达图像 (复用之前的逻辑)
        td, tr, ta = self._load_images(item)
        wave_tensor = self._process_wave(td, tr, ta)

        # 2. 构建对话 (Instruction Tuning)
        # 你的 train.json 里的 texts_ground_truth 是答案
        ground_truth = str(item.get('texts_ground_truth', ["unknown"])[0])
        
        # 简单的对话模板
        conversation = [
            {"role": "user", "content": "<radar_image>\nDescribe the human activity in this radar signal."},
            {"role": "assistant", "content": ground_truth}
        ]
        
        return {
            'pixel_values': wave_tensor,
            'conversation': conversation
        }

    def _process_wave(self, td, tr, ta):
        # 简单的预处理逻辑
        def to_tensor(arr):
            arr = arr.astype(np.float32)
            if arr.max() - arr.min() > 1e-6:
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            t = torch.from_numpy(arr)
            if t.ndim == 2: t = t.unsqueeze(0)
            elif t.ndim == 3 and t.shape[2] == 1: t = t.permute(2, 0, 1)
            return F.interpolate(t.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        try:
            w = torch.cat([to_tensor(td), to_tensor(tr), to_tensor(ta)], dim=0)
            # Normalize (Important! Match Stage 1)
            w = (w - 0.5) / 0.5
            return w
        except:
            return torch.zeros((3, 224, 224), dtype=torch.float32)

    def _load_images(self, item):
        try:
            def safe_load(path_key):
                rel_path = item.get(path_key, '')
                fname = os.path.basename(rel_path)
                folder = 'imgs_test' if 'imgs_test' in rel_path else 'imgs'
                full_path = os.path.join(self.root_dir, folder, fname)
                return np.load(full_path)
            return safe_load('td_path'), safe_load('tr_path'), safe_load('ta_path')
        except:
            z = np.zeros((64, 64), dtype=np.float32)
            return z, z, z