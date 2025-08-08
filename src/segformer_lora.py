import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from peft import LoraConfig, get_peft_model

class SegFormerWithLoRA(nn.Module):
    def __init__(self, num_classes=3, img_size=512, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        cfg = SegformerConfig(
            num_channels=3,
            num_labels=num_classes,
            hidden_sizes=[64, 128, 320, 512],
            depths=[2, 2, 2, 2],
            decoder_hidden_size=256
        )
        self.backbone = SegformerModel(cfg)
        # Add simple decoder head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        # Apply LoRA to attention projections in encoder
        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"], bias="none", task_type="SEQ_CLS"
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)

    def forward(self, x):
        # SegformerModel outputs feature maps in hidden_states; use last stage
        outputs = self.backbone(x, output_hidden_states=True)
        # last hidden state is B,C,H,W for segformer’s last stage (via reshaping in model)
        # depending on version, you might need to adapt the tensor shape
        feat = outputs.last_hidden_state  # (B, HW, C) or (B, C, H, W) per version
        if feat.dim() == 3:  # (B, HW, C) -> (B, C, H, W)
            B, HW, C = feat.shape
            H = W = int(HW ** 0.5)
            feat = feat.transpose(1, 2).contiguous().view(B, C, H, W)
        logits = self.head(feat)
        return logits
