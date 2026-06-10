import torch
import torch.nn as nn
import torchvision.models as models
from torchaudio.models import Conformer

# conf = Conformer(input_dim=256, num_heads=4, ffn_dim=256*4, num_layers=6, depthwise_conv_kernel_size=31)


class ResNet34Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet34 = models.resnet34(weights=None)
        resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet34.children())[:-1])  # backbone
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)  
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.squeeze(-1).squeeze(-1)   # flatten [B, 512]
        z = self.projection(h)          # projected embedding
        z = torch.nn.functional.normalize(z, dim=-1)
        return z  # h = encoder features, z = projection head output


class ConformerEncoder(nn.Module):
    def __init__(self, embedding_dim=256,
                  input_dim = 65,
                  conformer_input_dim=256,
                  num_layers=10,
                  num_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, conformer_input_dim)

        self.conformer = Conformer(input_dim=conformer_input_dim,
                                   num_heads=num_heads,
                                   ffn_dim=conformer_input_dim*4,
                                   num_layers=num_layers,
                                   depthwise_conv_kernel_size=31,
                                   dropout=0.1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding_projection = nn.Sequential(nn.Linear(conformer_input_dim, embedding_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(embedding_dim, embedding_dim))
    def forward(self, x, lengths=None):
        
        x = self.input_proj(x)

        x, lengths = self.conformer(x, lengths)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        emb = self.embedding_projection(x)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb
    
class simpleRegressor(nn.Module):
    def __init__(self, embedding_dim, neurons_per_layer = 256):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(embedding_dim, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU()
        )
        self.output_proj = nn.Linear(neurons_per_layer, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.output_proj(self.cls(x))
        return out
    
