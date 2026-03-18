import torch
import torch.nn as nn
import timm
from transformers import AutoModel
import numpy as np

class SimpleMMClip(nn.Module):
    def __init__(self, radar_backbone_name='vit_base_patch16_224', text_model_name='bert-base-uncased', embed_dim=512):
        super().__init__()
        
        # 1. Radar Encoder (使用 timm 加载 ViT)
        self.radar_encoder = timm.create_model(radar_backbone_name, pretrained=True, num_classes=0) 
        radar_feature_dim = self.radar_encoder.num_features
        
        # 投影层
        self.radar_projection = nn.Linear(radar_feature_dim, embed_dim)
        
        # 2. Text Encoder (使用 HuggingFace BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_feature_dim = self.text_encoder.config.hidden_size
        
        # 投影层
        self.text_projection = nn.Linear(text_feature_dim, embed_dim)
        
        # 3. 温度系数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, pixel_values, input_ids, attention_mask):
        # Radar Branch
        radar_features = self.radar_encoder(pixel_values) 
        radar_embeds = self.radar_projection(radar_features)
        
        # Text Branch
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] 
        text_embeds = self.text_projection(text_features)
        
        # Normalize
        radar_embeds = radar_embeds / radar_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        
        return radar_embeds, text_embeds, self.logit_scale.exp()

def contrastive_loss(radar_embeds, text_embeds, logit_scale):
    logits_per_radar = logit_scale * radar_embeds @ text_embeds.t()
    logits_per_text = logits_per_radar.t()
    
    batch_size = radar_embeds.shape[0]
    labels = torch.arange(batch_size, device=radar_embeds.device)
    
    loss_r = nn.functional.cross_entropy(logits_per_radar, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    
    return (loss_r + loss_t) / 2