import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadAttributeClassifier(nn.Module):
    def __init__(self, output_dims):
        """
        output_dims: dict, e.g., {'action_category': 5, 'posture': 3, ...}
        """
        super(MultiHeadAttributeClassifier, self).__init__()
        
        # 1. 骨干网络 (ResNet18)
        # 使用 ImageNet 预训练权重加速收敛
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 输入已经是3通道(TD,TR,TA)，无需修改输入层
        # 去掉最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        feature_dim = resnet.fc.in_features # ResNet18 是 512
        
        # 2. 定义多个预测头 (Prediction Heads)
        self.heads = nn.ModuleDict()
        for attr_name, num_classes in output_dims.items():
            self.heads[attr_name] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                # 优化: Dropout 降低到 0.3，避免在多头任务中丢失过多信息
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        # x shape: [Batch, 3, H, W]
        # 提取共享特征
        features = self.backbone(x) # [Batch, 512, 1, 1]
        
        # 每个头独立预测
        outputs = {}
        for attr_name, head in self.heads.items():
            outputs[attr_name] = head(features)
            
        return outputs