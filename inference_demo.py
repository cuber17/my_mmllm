import torch
import torch.nn.functional as F
import numpy as np
import os
import timm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.attributes_perception.inference_utils import AttributePredictor
from src.llm.projector import RadarProjector
import json

class MMExpertInference:
    def __init__(self, 
                 radar_encoder_path, 
                 projector_path, 
                 attr_model_path, 
                 attr_label_map,
                 llm_base_path, 
                 llm_adapter_path, 
                 data_json_path,  # 新增: 数据集JSON路径
                 data_root_dir,   # 新增: 数据集根目录
                 device="cuda"):
        
        self.device = device
        self.data_root = data_root_dir
        print(f">>> Initializing MMExpert on {device}...")

        # 0. 加载数据索引
        print(f"Loading dataset index from {data_json_path}...")
        with open(data_json_path, 'r') as f:
            data_list = json.load(f)
        # 构建 ID -> Item 的映射，方便快速查找
        self.data_index = {item['id']: item for item in data_list}

        # A. 加载属性识别头 (Attribute Head)
        print("Loading Attribute Predictor...")
        self.attr_predictor = AttributePredictor(attr_model_path, attr_label_map, device=device)

        # B. 加载 Radar Encoder (CLIP Visual Encoder)
        print("Loading Radar Encoder...")
        self.radar_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        # 加载 clean 过的权重 (去除 radar_encoder. 前缀)
        ckpt = torch.load(radar_encoder_path, map_location='cpu')
        new_ckpt = {k.replace("radar_encoder.", ""): v for k, v in ckpt.items()}
        self.radar_encoder.load_state_dict(new_ckpt, strict=False)
        self.radar_encoder.to(device).eval()

        # C. 加载 Projector
        print("Loading Projector...")
        # 假设维度：ViT Base 768 -> Phi-3 3072
        self.projector = RadarProjector(encoder_dim=768, llm_dim=3072).to(device)
        self.projector.load_state_dict(torch.load(projector_path, map_location=device))
        self.projector.eval()

        # D. 加载 LLM & LoRA
        print("Loading LLM (Phi-3) & LoRA...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_base_path, trust_remote_code=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            llm_base_path, 
            trust_remote_code=False, 
            dtype=torch.bfloat16,
            device_map=device
        )
        self.llm = PeftModel.from_pretrained(base_model, llm_adapter_path)
        self.llm.eval()
        
        print(">>> Model Loading Complete.")

    def _resolve_path(self, path):
        """处理相对路径问题"""
        if path.startswith('.'):
            # 去掉 ./ 并拼接到 data_root
            return os.path.join(self.data_root, path.lstrip('./'))
        return path

    def preprocess_radar(self, td_path, tr_path, ta_path):
        """读取三个热图npy，归一化并Resize到224x224"""
        def load_and_norm(path):
            arr = np.load(path).astype(np.float32)
            # Min-Max Normalize to [0, 1]
            if arr.max() - arr.min() > 1e-6:
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            return torch.from_numpy(arr)

        td = load_and_norm(td_path)
        tr = load_and_norm(tr_path)
        ta = load_and_norm(ta_path)

        # H, W -> 1, H, W
        if td.ndim == 2: td = td.unsqueeze(0)
        if tr.ndim == 2: tr = tr.unsqueeze(0)
        if ta.ndim == 2: ta = ta.unsqueeze(0)

        # 处理通道维度问题 (部分数据可能不一致)
        # 统一 Resize
        target_size = (224, 224)
        def resize_t(t):
            # Interp 需要 4D 输入 (B, C, H, W)
            return F.interpolate(t.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        td = resize_t(td)
        tr = resize_t(tr)
        ta = resize_t(ta)

        # Stack -> (3, 224, 224)
        img_tensor = torch.cat([td, tr, ta], dim=0)
        return img_tensor.unsqueeze(0) # (1, 3, 224, 224)

    def generate(self, sample_id, question="Describe the human activity.", attr_threshold=0.75): # [修改] 默认阈值设为 0.75
        # [修改] 接受 sample_id
        if sample_id not in self.data_index:
            raise ValueError(f"Sample ID {sample_id} not found in the loaded index.")

        item = self.data_index[sample_id]
        td_path = self._resolve_path(item['td_path'])
        tr_path = self._resolve_path(item['tr_path'])
        ta_path = self._resolve_path(item['ta_path'])

        # 检查文件是否存在
        if not os.path.exists(td_path):
            print(f"Warning: File not found at {td_path}, trying simplified path...")
            # Fallback (针对可能的路径不一致)
            fname = os.path.basename(item['td_path'])
            td_path = os.path.join(self.data_root, "imgs_test", fname)
            tr_path = td_path.replace("td", "tr")
            ta_path = td_path.replace("td", "ta")

        # 1. 预处理数据
        radar_tensor = self.preprocess_radar(td_path, tr_path, ta_path) # [0, 1] range
        
        # 2. 路径二：获取显式属性 (Explicit Attributes)
        # AttributePredictor 内部需要 Normalize 到 [-1, 1]，它自己会处理 transforms
        with torch.no_grad():
            attr_input = radar_tensor.clone().to(self.device).float()
            # [修改] 传入置信度阈值
            attr_res, attr_prompt = self.attr_predictor.predict_attributes(attr_input.squeeze(0), threshold=attr_threshold)

        # 3. 路径一：获取隐式语义 (Implicit Embeddings)
        radar_tensor = radar_tensor.to(self.device).float() # ViT 输入
        with torch.no_grad():
            radar_feats = self.radar_encoder.forward_features(radar_tensor) # [1, 197, 768]
            radar_embeds = self.projector(radar_feats).to(dtype=torch.bfloat16) # [1, 197, 3072]

        # 4. 构造 Prompt
        # [修改] 优化 Prompt 结构：如果 attr_prompt 不是空的，才加前缀
        if "Action:" in attr_prompt or "Posture:" in attr_prompt:
             full_user_prompt = f"Observed attributes with high confidence: {attr_prompt} {question}"
        else:
             # 如果所有属性都被过滤掉了，只依靠 Embedding
             full_user_prompt = f"{question}"
        
        conversation = [{"role": "user", "content": full_user_prompt}]
        text_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        tokens = self.tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # 5. 拼接 Embedding: [Radar, Text]
        text_embeds = self.llm.model.model.embed_tokens(tokens.input_ids)
        inputs_embeds = torch.cat([radar_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([
            torch.ones(1, radar_embeds.shape[1], device=self.device),
            tokens.attention_mask
        ], dim=1)

        # 6. 生成
        # 设定终止符：Phi-3 有时使用 <|end|> 或 <|endoftext|>
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            self.tokenizer.convert_tokens_to_ids("<|end|>")
        ]
        
        with torch.no_grad():
            generate_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        # 解码
        response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        return response, attr_prompt, item.get('texts_ground_truth', ['Unknown'])[0]


if __name__ == "__main__":
    # --- 配置路径 ---
    BASE_DIR = "/root/jyz/my_mmLLM"
    
    # 1. Attribute Model
    ATTR_EXP_ID = "experiment_20260118_185816" 
    ATTR_CKPT = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/best.pth"
    ATTR_MAP = f"{BASE_DIR}/logs/{ATTR_EXP_ID}/label_maps.json"
    
    # 2. Radar Encoder (Stage 1)
    RADAR_CKPT = f"{BASE_DIR}/logs/clip_20260120_224659/radar_encoder_only.pth"
    
    # 3. Projector & LLM Adapter (Stage 2)
    STAGE2_EPOCH = "epoch_2"
    STAGE2_DIR = f"{BASE_DIR}/logs/stage2_20260121_210018/{STAGE2_EPOCH}"
    PROJ_CKPT = f"{STAGE2_DIR}/projector.pth"
    LORA_PATH = f"{STAGE2_DIR}"
    
    # 4. LLM Base
    LLM_BASE = f"{BASE_DIR}/huggingface/Phi-3-mini-4k-instruct"

    # 5. Dataset Config
    DATA_JSON = f"{BASE_DIR}/processed_dataset/test.json"
    DATA_ROOT = f"{BASE_DIR}/processed_dataset/"

    # --- 实例化 ---
    try:
        model = MMExpertInference(
            radar_encoder_path=RADAR_CKPT,
            projector_path=PROJ_CKPT,
            attr_model_path=ATTR_CKPT,
            attr_label_map=ATTR_MAP,
            llm_base_path=LLM_BASE,
            llm_adapter_path=LORA_PATH,
            data_json_path=DATA_JSON,  # Pass new args
            data_root_dir=DATA_ROOT
        )

        # --- 连续推理循环 ---
        print("\n" + "="*50)
        print("MMExpert Inference - Interactive Mode")
        print("Type 'q', 'quit', or 'exit' to stop.")
        print("="*50)
        
        while True:
            # 获取用户输入
            user_input = input("\nEnter Sample ID to Test: ").strip()
            
            # 检查退出条件
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting inference demo. Goodbye!")
                break
            
            # 跳过空输入
            if not user_input:
                continue

            try:
                print(f"Testing ID: {user_input}")
                response, detected_attrs, gt = model.generate(user_input)
                
                print(f"\n[Detected Attributes]: {detected_attrs}")
                print(f"[Ground Truth]: {gt}")
                print(f"[LLM Generated Description]:\n{response}")
                print("-" * 50)
                
            except ValueError as ve:
                print(f"Instance Error: {ve}")
                print("Please try another ID.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"An unexpected error occurred during generation: {e}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFatal Error initializing model: {e}")