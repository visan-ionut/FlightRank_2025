import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_embedding_dim(cardinality: int, min_dim=2, max_dim=32) -> int:
    if cardinality <= 3:
        return 1
    else:
        dim = round(4 * math.log2(cardinality + 1))
        dim = max(dim, min_dim)
        dim = min(dim, max_dim)
        return dim


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, N, C]
        w = self.fc(x.mean(dim=1))
        w = w.unsqueeze(1)  # [B, 1, C]
        return x * w


class DLRankerAttention(nn.Module):
    def __init__(
        self,
        cat_dims,
        num_numeric_feats,
        emb_dropout=0.1,
        dropout=0.1,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        proj_dim=64,
    ):
        super().__init__()

        self.emb_dims = {f: get_embedding_dim(cat_dims[f]) for f in cat_dims}

        print("Embedding dims:")
        for f, d in self.emb_dims.items():
            print(f"{f}: {d}")

        self.embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(cat_dims[f], self.emb_dims[f], padding_idx=0)
                for f in cat_dims
            }
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        total_emb_dim = sum(self.emb_dims.values())
        print(
            f"Categorial emb dim sum: {total_emb_dim}, Numerical dim: {num_numeric_feats}"
        )

        self.cat_proj = nn.Linear(total_emb_dim, proj_dim)
        self.cat_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.num_proj_in = nn.Linear(num_numeric_feats, proj_dim)
        self.num_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.num_se = SEModule(proj_dim)
        self.num_proj_out = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=2 * proj_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(2 * proj_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat: dict, x_num: torch.Tensor, attn_mask=None):
        B, N, _ = x_num.shape  # x_num: [B, N, F_num]

        cat_emb_list = []
        for f, emb_layer in self.embeddings.items():
            emb = emb_layer(x_cat[f])  # [B, N]
            cat_emb_list.append(emb)
        cat_embs = torch.cat(cat_emb_list, dim=-1)  # [B, N, total_emb_dim]
        cat_embs = self.emb_dropout(cat_embs)
        cat_feat = self.cat_proj(cat_embs)  # [B, N, proj_dim]
        cat_feat = self.cat_transformer(cat_feat)  # [B, N, proj_dim]

        num_feat = self.num_proj_in(x_num)  # [B, N, proj_dim]
        num_feat = self.num_transformer(num_feat)  # [B, N, proj_dim]
        num_feat = self.num_se(num_feat)  # [B, N, proj_dim]
        num_feat = self.num_proj_out(num_feat)  # [B, N, proj_dim]

        x = torch.cat([cat_feat, num_feat], dim=-1)  # [B, N, 2*proj_dim]

        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=attn_mask)

        scores = self.mlp(x).squeeze(-1)  # [B, N]

        return scores


class FiLM(nn.Module):
    def __init__(self, cat_dim, num_feat_dim):
        super().__init__()
        self.scale_gen = nn.Linear(cat_dim, num_feat_dim)
        self.shift_gen = nn.Linear(cat_dim, num_feat_dim)

        self.attn_w = nn.Linear(cat_dim, 1)

    def forward(self, cat_feat, num_feat):
        # cat_feat: [B, G, C_cat], num_feat: [B, G, C_num]

        # 计算加权注意力权重，先映射，后softmax归一化
        attn_scores = self.attn_w(cat_feat)  # [B, G, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # 按组内元素归一化

        # 加权池化
        cat_weighted = torch.sum(attn_weights * cat_feat, dim=1)  # [B, C_cat]

        # 生成 FiLM 参数
        scale = self.scale_gen(cat_weighted).unsqueeze(1)  # [B, 1, C_num]
        shift = self.shift_gen(cat_weighted).unsqueeze(1)  # [B, 1, C_num]

        # 调制数值特征
        modulated_num_feat = num_feat * scale + shift
        return modulated_num_feat


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-Norm Attention + Residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-Norm FFN + Residual
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.kernels = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )
        self.lns = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    def forward(self, x):
        x_0 = x
        x_l = x
        for i in range(self.num_layers):
            xl_w = self.kernels[i](x_l)
            cross = x_0 * xl_w
            x_l = cross + self.bias[i] + x_l
            x_l = self.lns[i](x_l)
            x_l = F.silu(x_l)
            x_l = self.dropouts[i](x_l)
        return x_l


class DLRankerDeepTransformer(nn.Module):
    def __init__(
        self,
        cat_dims,
        num_numeric_feats,
        attn_dim=128,
        num_heads=4,
        num_layers=4,
        cross_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.emb_dims = {f: get_embedding_dim(cat_dims[f]) for f in cat_dims}
        self.embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(cat_dims[f], self.emb_dims[f], padding_idx=0)
                for f in cat_dims
            }
        )

        total_emb_dim = sum(self.emb_dims.values())
        print(
            f"Categorial emb dim sum: {total_emb_dim}, Numerical dim: {num_numeric_feats}"
        )

        self.num_ln = nn.LayerNorm(num_numeric_feats)
        self.input_proj = nn.Linear(total_emb_dim + num_numeric_feats, attn_dim)

        # 堆叠 Transformer 层
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(attn_dim, num_heads, attn_dim * 2, dropout)
                for _ in range(num_layers)
            ]
        )

        # Cross Net
        self.crossnet = CrossNet(attn_dim, cross_layers)

        # Scale/Shift
        self.film = FiLM(total_emb_dim, num_numeric_feats)
        self.se_layer = SEModule(attn_dim)

        # MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_num):
        B, G = x_num.shape[:2]

        # Layernorm for numerical feature
        x_num = x_num.view(-1, x_num.shape[-1])
        x_num = self.num_ln(x_num)
        x_num = x_num.view(B, G, -1)

        # Embedding for categorial feature
        emb_list = [self.embeddings[f](x_cat[f]) for f in self.embeddings]
        x_cat_emb = torch.cat(emb_list, dim=-1)

        x_num = self.film(x_cat_emb, x_num)

        # Concate and proj
        x = torch.cat([x_cat_emb, x_num], dim=-1)
        x = self.input_proj(x)

        # Transformer
        for layer in self.transformer_layers:
            x = layer(x)

        # Cross Net
        x_cross = self.crossnet(x)
        x = x + x_cross

        x = self.se_layer(x)

        # MLP
        scores = self.output_mlp(x).squeeze(-1)  # [B, G]
        return scores


class DLRanker(nn.Module):
    def __init__(
        self, cat_dims, num_numeric_feats, emb_dim=64, hidden=[512, 256, 128, 64]
    ):
        super().__init__()

        self.embeddings = nn.ModuleDict(
            {f: nn.Embedding(cat_dims[f], emb_dim, padding_idx=0) for f in cat_dims}
        )
        self.linear_cat = nn.ModuleDict(
            {f: nn.Embedding(cat_dims[f], 1, padding_idx=0) for f in cat_dims}
        )

        for emb in list(self.embeddings.values()) + list(self.linear_cat.values()):
            nn.init.xavier_uniform_(emb.weight)

        self.linear_num = nn.Linear(num_numeric_feats, 1)
        nn.init.xavier_uniform_(self.linear_num.weight)

        input_dim = emb_dim * len(cat_dims) + num_numeric_feats
        layers = []
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x_cat, x_num):
        batch_size, group_size = x_num.shape[:2]

        linear_cat_sum = torch.stack(
            [self.linear_cat[f](x_cat[f]).squeeze(-1) for f in self.linear_cat], dim=0
        ).sum(dim=0)

        linear_num_out = self.linear_num(x_num).squeeze(-1)
        linear_out = linear_cat_sum + linear_num_out

        embs = torch.stack(
            [self.embeddings[f](x_cat[f]) for f in self.embeddings], dim=2
        )

        sum_emb = embs.sum(dim=2)
        sum_emb_square = sum_emb**2
        square_emb_sum = (embs**2).sum(dim=2)
        fm_out = 0.0
        # fm_out = 0.5 * (sum_emb_square - square_emb_sum).sum(dim=2)

        embs_cat = embs.reshape(batch_size * group_size, -1)
        x_num_flat = x_num.reshape(batch_size * group_size, -1)
        deep_input = torch.cat([embs_cat, x_num_flat], dim=1)
        deep_out = self.mlp(deep_input)
        deep_out = self.output(deep_out).view(batch_size, group_size)

        return linear_out + fm_out + deep_out
