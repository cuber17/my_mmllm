import os
from huggingface_hub import snapshot_download

def download_phi3():
    # 目标路径
    save_dir = "./huggingface/Phi-3-mini-4k-instruct"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Start downloading Phi-3 model to {save_dir}...")
    
    # 使用镜像加速下载 (如果服务器连不上HF官网)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 忽略非 PyTorch 权重
    )
    print("Download finished!")

if __name__ == "__main__":
    download_phi3()