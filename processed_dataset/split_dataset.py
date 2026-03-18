import json
import os
import random
import shutil
from tqdm import tqdm

def split_dataset():
    # 配置路径
    base_dir = '/root/jyz/my_mmLLM/processed_dataset'
    train_json_path = os.path.join(base_dir, 'train.json')
    test_json_path = os.path.join(base_dir, 'test.json')
    
    # 图片源目录和目标目录
    source_img_dir = os.path.join(base_dir, 'imgs') # 假设原图都在这里
    target_img_dir = os.path.join(base_dir, 'imgs_test') # 测试集图片放这里

    # 分割数量
    test_size = 1000

    # 1. 检查并创建目录
    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir)
        print(f"Created directory: {target_img_dir}")
    else:
        print(f"Directory exists: {target_img_dir}")

    # 2. 读取 Json 数据
    print(f"Loading {train_json_path}...")
    with open(train_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_len = len(data)
    print(f"Total samples: {total_len}")
    
    if total_len < test_size:
        print("Error: Dataset size is smaller than requested split size.")
        return

    # 3. 随机打乱并切分
    # 为了可复现，可以设置随机种子
    random.seed(42)
    random.shuffle(data)

    test_data_raw = data[:test_size]
    train_data_raw = data[test_size:]

    print(f"Splitting: Train={len(train_data_raw)}, Test={len(test_data_raw)}")

    # 4. 处理测试集数据：移动文件并更新路径
    test_data_processed = []
    
    print("Processing test data (moving files)...")
    for item in tqdm(test_data_raw):
        new_item = item.copy()
        
        # 需要处理的三个键
        path_keys = ['td_path', 'tr_path', 'ta_path']
        
        for key in path_keys:
            # 原始路径例如: "./imgs/000000_td.npy"
            original_rel_path = item[key]
            
            # 提取文件名: "000000_td.npy"
            file_name = os.path.basename(original_rel_path)
            
            # 构造绝对源路径
            # 注意：原始json里的路径是相对路径，需要小心拼接
            # 如果json里是 "./imgs/xxx"，我们需要去掉 "./imgs/" 或者直接用basename
            # 这里假设源文件就在 base_dir/imgs/ 下
            src_file = os.path.join(source_img_dir, file_name)
            dst_file = os.path.join(target_img_dir, file_name)
            
            # 移动文件
            try:
                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)
                else:
                    # 尝试检查文件是否已经被移动过(防止重复运行脚本报错)
                    if not os.path.exists(dst_file):
                        print(f"Warning: Source file not found {src_file}")
            except Exception as e:
                print(f"Error moving {src_file}: {e}")

            # 更新 JSON 中的路径
            # 新路径应为 "./imgs_test/000000_td.npy"
            new_item[key] = f"./imgs_test/{file_name}"
        
        test_data_processed.append(new_item)

    # 5. 保存文件
    print(f"Saving new train.json with {len(train_data_raw)} samples...")
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data_raw, f, indent=4)

    print(f"Saving test.json with {len(test_data_processed)} samples...")
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data_processed, f, indent=4)

    print("Done! Split completed.")

if __name__ == "__main__":
    split_dataset()