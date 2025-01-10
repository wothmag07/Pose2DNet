import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(energy, dim=-1)

        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)

        return self.fc_out(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.self_attn(x, x, x)
        x = x + self.dropout(self.norm1(attention))
        mlp_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(self.norm2(mlp_output))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_dim, num_classes, num_keypoints, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size ** 2

        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1 + num_keypoints, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        # Add the cls_token to the input
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :(num_patches + 1)]

        x = self.dropout(x)

        for layer in self.transformer_encoder_layers:
            x = layer(x)

        x = self.mlp_head(x[:, 0])

        return x

class PoseEstimationModel(nn.Module):
    def __init__(self, image_size, num_keypoints, patch_size=16, hidden_size=512, num_layers=6, num_heads=8, mlp_dim=2048):
        super(PoseEstimationModel, self).__init__()
        self.image_size = image_size
        self.num_keypoints = num_keypoints

        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * 3, hidden_size)
        )

        self.transformer = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=hidden_size,
            embed_dim=hidden_size,
            depth=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_classes=self.num_keypoints * 2,   # (2D keypoints)
            num_keypoints=self.num_keypoints
        )

    def forward(self, images, keypoints):
        patches = self.patch_embed(images)

        # Create positional embeddings for keypoints
        keypoints_pos_emb = self._create_positional_embeddings(keypoints, patches.device)

        # Concatenate patches with positional embeddings for keypoints
        embeddings = torch.cat([patches, keypoints_pos_emb], dim=1)

        keypoints_pred = self.transformer(embeddings)
        keypoints_pred = keypoints_pred.view(keypoints_pred.size(0), self.num_keypoints, 2)

        return keypoints_pred

    def _create_positional_embeddings(self, keypoints, device):
        batch_size = keypoints.shape[0]
        num_keypoints = keypoints.shape[1]
        embedding_dim = self.transformer.pos_embedding.shape[-1]
        
        # Rescale keypoints coordinates to fit the image size
        keypoints = keypoints.float() / self.image_size

        # Create positional embeddings for keypoints
        keypoints_pos_emb = torch.zeros(batch_size, num_keypoints, embedding_dim, device=device)
        for i in range(batch_size):
            for j in range(num_keypoints):
                x, y = keypoints[i, j]
                pos_emb = torch.sin(torch.arange(0, embedding_dim, 2, device=device) * -(math.log(10000.0) / embedding_dim))
                keypoints_pos_emb[i, j, 0::2] = torch.sin(x * pos_emb)
                keypoints_pos_emb[i, j, 1::2] = torch.cos(y * pos_emb)
        
        return keypoints_pos_emb

