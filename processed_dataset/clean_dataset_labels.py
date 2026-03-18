import json
import os
import re

# 1. 定义标准标签空间 (白名单)
VALID_LABELS = {
    'action_category': ['locomotion', 'stationary_activity', 'gesture', 'exercise', 'transition'],
    'posture': ['upright', 'sitting', 'crouching', 'lying', 'bending'],
    'intensity': ['static', 'slow', 'normal', 'vigorous'],
    'active_part': ['full_body', 'upper_body', 'lower_body', 'head_neck'],
    'trajectory': ['in_place', 'forwards', 'backwards', 'lateral_move', 'dynamic_turn']
}

# 2. 定义默认值 (当完全无法匹配时的保底值)
# 根据你的数据集特性，这里选择出现频率最高的词作为默认值
DEFAULT_LABELS = {
    'action_category': 'gesture',
    'posture': 'upright',
    'intensity': 'normal',
    'active_part': 'full_body', 
    'trajectory': 'in_place'
}

def clean_single_label(key, raw_value, valid_set):
    """
    清洗单个标签值的逻辑
    """
    if not isinstance(raw_value, str):
        return DEFAULT_LABELS[key]

    # 预处理：转小写，去除首尾空格
    clean_val = raw_value.lower().strip()
    
    # 策略 1: 直接在白名单中
    if clean_val in valid_set:
        return clean_val
    
    # 策略 2: 模糊匹配 (白名单中的词是否包含在当前杂乱文本中)
    # 优先匹配较长的词，防止误判
    # 例如：'slowly' contains 'slow' -> 'slow'
    sorted_candidates = sorted(valid_set, key=len, reverse=True)
    for candidate in sorted_candidates:
        if candidate in clean_val:
            return candidate
            
    # 策略 3: 手动纠错规则 (针对常见的 dirty patterns)
    # 针对 Posture 的特殊修正
    if key == 'posture':
        if 'stand' in clean_val: return 'upright'
        if 'kneel' in clean_val: return 'crouching'
        if 'squat' in clean_val: return 'crouching'
        if 'bow' in clean_val: return 'bending'
        
    # 针对 Trajectory 的特殊修正
    if key == 'trajectory':
        if 'forward' in clean_val: return 'forwards'  # typo fix
        if 'backward' in clean_val: return 'backwards'
        if 'left' in clean_val or 'right' in clean_val: return 'lateral_move' 
        if 'circle' in clean_val or 'turn' in clean_val: return 'dynamic_turn'
        if 'nil' in clean_val: return 'in_place'

    # 针对 Intensity 的特殊修正
    if key == 'intensity':
        if 'fast' in clean_val: return 'vigorous'
        
    # 策略 4: 完全无法识别，使用默认值
    print(f"  [Warning] Key: {key} | Unfixable value: '{raw_value}' -> Fallback: '{DEFAULT_LABELS[key]}'")
    return DEFAULT_LABELS[key]

def process_file(input_path, output_path):
    print(f"Processing {input_path} ...")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    fixed_count = 0
    
    for item in data:
        labels = item.get('labels', {})
        new_labels = {}
        is_modified = False
        
        for key, valid_set in VALID_LABELS.items():
            raw_val = labels.get(key, "")
            cleaned_val = clean_single_label(key, raw_val, valid_set)
            
            new_labels[key] = cleaned_val
            
            if cleaned_val != raw_val:
                is_modified = True
        
        # 更新 item
        item['labels'] = new_labels
        if is_modified:
            fixed_count += 1

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"Finished. Total: {total}, Fixed/Normalized: {fixed_count}")
    print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    # 配置你的实际路径
    base_dir = "/root/jyz/my_mmLLM/processed_dataset"
    
    files_to_clean = [
        ("train.json", "train_cleaned.json"),
        ("test.json", "test_cleaned.json")
    ]
    
    for in_file, out_file in files_to_clean:
        in_path = os.path.join(base_dir, in_file)
        out_path = os.path.join(base_dir, out_file)
        process_file(in_path, out_path)