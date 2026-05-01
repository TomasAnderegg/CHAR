import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class TemporalAttn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, gru_outputs):
        # gru_outputs: (B, 21, 128)
        scores = self.attn(gru_outputs).squeeze(-1)   # (B, 21)
        weights = F.softmax(scores, dim=1).unsqueeze(1)  # (B, 1, 21)
        context = torch.bmm(weights, gru_outputs).squeeze(1)  # (B, 128)
        return context

class DrivingPlanner(nn.Module):
    """
    LTF-inspired architecture:
    ResNet-34 (image) + Command Embedding + History GRU → Fusion MLP → GRU Decoder → Trajectory (60, 3)
    """
    def __init__(
        self,
        num_commands=3,
        command_embed_dim=64,
        history_input_dim=4,   # 4 (x,y,sin, cos) + 9 (dynamics) si include_dynamics=True, sinon 3
        history_hidden=128,
        fusion_dim=512,
        gru_hidden=512,
        output_steps=60,
        output_dim=3,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.output_dim = output_dim

        # ── 1. Image encoder : ResNet-34 pretrained ──────────────────
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # ── 2. Driving command embedding ──────────────────────────────
        self.command_embed = nn.Embedding(num_commands, command_embed_dim)

        # ── 3. History encoder : GRU ──────────────────────────────────
        # history_input_dim = 3 sans dynamics, 8 avec
        self.history_gru = nn.GRU(
            input_size=history_input_dim,
            hidden_size= 256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # ── 4. Fusion MLP ─────────────────────────────────────────────
        # 256 (image) + 64 (cmd) + 256 (history) = 448
        self.fusion = nn.Sequential(
            nn.Linear(256 + command_embed_dim + 256, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, gru_hidden),
            nn.ReLU(),
        )

        # ── 5. Trajectory decoder : MLP direct ────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gru_hidden, output_steps * output_dim),
        )

        # --6. Temporal Attention---------------------------------------
        self.history_attn = TemporalAttn(256)#history_hidden)

    def forward(self, image, command, history):
        B = image.size(0)

        # Image
        feat = self.image_encoder(image)          # (B, 512, h, w)
        feat = self.image_pool(feat).flatten(1)   # (B, 512)
        img_feat = self.image_proj(feat)          # (B, 256)

        # Command
        cmd_feat = self.command_embed(command)    # (B, 64)

        # History
        gru_out, _ = self.history_gru(history)   # (B, 21, 128)
        hist_feat = self.history_attn(gru_out)   # (B, 128)

        # Fusion
        combined = torch.cat([img_feat, cmd_feat, hist_feat], dim=1)  # (B, 448)
        hidden = self.fusion(combined)            # (B, 512)

        # Décodage direct — prédit les 60 deltas en une passe, cumsum depuis le dernier point
        start = history[:, -1, :self.output_dim].unsqueeze(1)          # (B, 1, 3)
        deltas = self.decoder(hidden).view(B, self.output_steps, self.output_dim)  # (B, 60, 3)
        return start + torch.cumsum(deltas, dim=1)                      # (B, 60, 3)
