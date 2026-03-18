import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class MMWaveAttributeDataset(Dataset):
    def __init__(self, json_file, root_dir, phase='train', transform=None, label_maps=None):
        self.root_dir = root_dir
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # 这里的 transform 将在 stack 和 resize 之后应用 (例如 Normalize)
        self.transform = transform
        
        # 定义标准标签空间 (白名单)
        self.valid_labels = {
            'action_category': ['locomotion', 'stationary_activity', 'gesture', 'exercise', 'transition'],
            'posture': ['upright', 'sitting', 'crouching', 'lying', 'bending'],
            'intensity': ['static', 'slow', 'normal', 'vigorous'],
            'active_part': ['full_body', 'upper_body', 'lower_body', 'head_neck'],
            'trajectory': ['in_place', 'forwards', 'backwards', 'lateral_move', 'dynamic_turn']
        }
        
        if label_maps is None:
            self.label_maps = self._build_standard_label_maps()
        else:
            self.label_maps = label_maps

    def _build_standard_label_maps(self):
        return {k: {label: idx for idx, label in enumerate(v)} for k, v in self.valid_labels.items()}

    def _clean_label(self, key, raw_label):
        valid_set = self.valid_labels[key]
        if not isinstance(raw_label, str): raw_label = ""
        raw_label = raw_label.lower().strip()
        
        if raw_label in valid_set:
            return raw_label
            
        for valid in valid_set:
            if valid in raw_label:
                return valid
                
        return valid_set[0] # Fallback

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 加载图片
        td, tr, ta = self._load_images(item)

        # 2. 预处理：转 Tensor + Resize 到 224x224
        # Dataset 负责确保输出尺寸统一
        target_size = (224, 224)
        
        def process_img(arr):
            arr = arr.astype(np.float32)
            # Min-Max 归一化到 [0, 1]
            if arr.max() - arr.min() > 1e-6:
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            
            t = torch.from_numpy(arr) 
            if t.ndim == 2: t = t.unsqueeze(0)
            elif t.ndim == 3 and t.shape[2] == 1: t = t.permute(2, 0, 1)
                
            # 使用双线性插值统一尺寸
            t = F.interpolate(t.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            return t

        try:
            td_t = process_img(td)
            tr_t = process_img(tr)
            ta_t = process_img(ta)
            # 堆叠 -> (3, 224, 224)
            img_tensor = torch.cat([td_t, tr_t, ta_t], dim=0)
        except Exception as e:
            print(f"Error processing tensor for {item.get('id')}: {e}")
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 3. 额外的 Transform (通常是 Normalize mean/std)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # 4. 处理标签
        label_tensors = {}
        for k in self.valid_labels.keys():
            raw_label = item['labels'].get(k, "")
            clean_lbl = self._clean_label(k, raw_label)
            int_label = self.label_maps[k][clean_lbl]
            label_tensors[k] = torch.tensor(int_label, dtype=torch.long)

        return img_tensor, label_tensors

    def _load_images(self, item):
        try:
            # [修复]: 增强的路径加载逻辑
            # 会依次尝试直接路径和拼接路径，避免只依赖 basename 的硬编码逻辑
            paths = [item['td_path'], item['tr_path'], item['ta_path']]
            loaded_arrays = []

            for p in paths:
                # 尝试 1: 直接拼接 (最推荐，假设 json 里是相对 processed_dataset 的路径)
                full_path_1 = os.path.join(self.root_dir, p)
                
                # 尝试 2: 兼容旧逻辑 (假设 json 里有路径但文件在 imgs/imgs_test 平铺)
                fname = os.path.basename(p)
                folder = 'imgs_test' if 'imgs_test' in p else 'imgs'
                full_path_2 = os.path.join(self.root_dir, folder, fname)

                if os.path.exists(full_path_1):
                    loaded_arrays.append(np.load(full_path_1))
                elif os.path.exists(full_path_2):
                    loaded_arrays.append(np.load(full_path_2))
                else:
                    raise FileNotFoundError(f"Cannot find file. Tried:\n1. {full_path_1}\n2. {full_path_2}")

            return loaded_arrays[0], loaded_arrays[1], loaded_arrays[2]

        except Exception as e:
            # [CRITICAL]: 打印错误，否则全黑图会导致准确率骤降但程序不崩
            print(f"[Dataset Error] Failed to load images for ID {item.get('id')}: {e}")
            # 失败返回 64x64 黑图
            z = np.zeros((64, 64), dtype=np.float32)
            return z, z, z

# 测试代码
if __name__ == "__main__":
    # 记得修改这里的 json 路径为你实际存在的路径
    dataset = MMWaveAttributeDataset(
        json_file='/root/jyz/my_mmLLM/processed_dataset/train.json', 
        root_dir='/root/jyz/my_mmLLM/processed_dataset/'
    )
    
    print("Label Maps (Cleaned):")
    print(json.dumps(dataset.label_maps, indent=2))
    
    img, labels = dataset[0]
    print(f"Img tensor shape: {img.shape}") 
    print(f"Labels: {labels}")