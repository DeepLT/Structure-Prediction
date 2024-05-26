import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from invariant_point_attention import InvariantPointAttention
from rigid import *
from utils import get_rel_pos

class IPALayer:
    # Invariant Point Attention Module
    def __init__(self, node_dim, edge_dim):
        self.IPA = InvariantPointAttention(
            dim = node_dim,              # node_dim
            heads = 8,                   # number of attention heads
            scalar_key_dim = 16,         # scalar query-key dimension
            scalar_value_dim = 16,       # scalar value dimension
            point_key_dim = 4,           # point query-key dimension
            point_value_dim = 4,         # point value dimension
            pairwise_repr_dim = edge_dim # edge_dim, pairwise representation dimension
        )

    def forward(self, node_features, edge_features, rotations, translations, mask=None):
        return self.IPA(
            node_features,
            edge_features,
            rotations = rotations,
            translations = translations,
            mask = mask
        )
    
class QuaternionUpdate(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.to_correction = nn.Linear(node_dim, 7)  # 4 for quaternion, 3 for translation

    def forward(self, node_features, update_mask=None):
        # Predict quaternion and translation vector
        q, t = self.to_correction(node_features).chunk(2, dim=-1)

        if update_mask is not None:
            q = update_mask[:, None] * q
            t = update_mask[:, None] * t

        # Normalize quaternion
        norm = torch.norm(q, dim=-1, keepdim=True)
        q = q / norm

        # Convert quaternion to rotation matrix
        a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        R = torch.stack([
            a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c,
            2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b,
            2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2
        ], dim=-1).reshape(q.shape[0], q.shape[1], 3, 3)

        return R, t


class StructureUpdate(nn.Module):
    # node_dim : amino acid 종류 20개
    # embed_dim : embedding 차원

    def __init__(self, node_dim, embed_dim, dropout_ratio):
        super().__init__()
        self.IPA = IPALayer(node_dim, embed_dim)
        self.norm1 = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.LayerNorm(node_dim)
        )
        self.residual = nn.Sequential(
            nn.Linear(node_dim, 2 * node_dim),
            nn.ReLU(),
            nn.Linear(2 * node_dim, 2 * node_dim),
            nn.ReLU(),
            nn.Linear(2 * node_dim, node_dim)
        )
        self.norm2 = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.LayerNorm(node_dim)
        )
        self.quaternion = QuaternionUpdate(node_dim)

        # residual 초기화
        # with torch.no_grad():
        #     self.residual[-1].weight.fill_(0.0)
        #     self.residual[-1].bias.fill_(0.0)
    
    def forward(self, node_features, edge_features, rigid, mask=None):
        # out : node_feature
        out = self.IPA(node_features, edge_features, rigid, mask)
        out = self.norm1(out)
        out = self.residual(out)
        out = self.norm2(out)

        # update rigid backbone

        return out


class StructureModule(nn.Module):
    def __init__(self, node_dim, embed_dim, rel_pos_dim, n_layers, dropout_ratio):
        super().__init__()
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.rel_pos_dim = rel_pos_dim

        self.node_embedding = nn.Linear(node_dim, embed_dim) # node_embedding

        # edge_embedding
        # rel_pos_embedding과 차원 맞춤 & rigid 속성과 concat 위해(차원 1 추가) embed_dim보다 차원 1개 적게 설정
        self.edge_embedding = nn.Linear(2 * rel_pos_dim + 1, embed_dim-1) 

        # StructureUpdate(model) init
        self.Layers = nn.ModuleList([StructureUpdate(node_dim, embed_dim, dropout_ratio)] for _ in n_layers)


    
    def forward(self, node_features, sequence):
        # node_features : (seq_len, nb_classes=20)
        # sequence : string type

        rigid = init_rigid(len(sequence), self.rel_pos_dim) # 초기화된 Rigid 생성
        
        # relative position calc.
        rel_pos = get_rel_pos(sequence) # rel_pos : (seq_len, seq_len),  값은 (0, 2 * rel_pos_dim) 의 범위 가짐

        # relative position embedding
        # num_classes=2*rel_pos_dim-1 : rel_pos의 값이 0~2*rel_pos_dim 사이이므로
        # rel_pos_embedding : (seq_len, seq_len, 2 * self.rel_pos_dim + 1)
        rel_pos_embedding = F.one_hot(rel_pos, num_classes=2 * self.rel_pos_dim + 1).to(dtype=node_features.dtype)

        rel_pos_embedding = self.edge_embedding(rel_pos_embedding) # rel_pos_embedding : (seq_len, seq_len, embed_dim - 1)

        # node_feature embedding
        node_features = self.node_embedding(node_features) # node_features : (seq_len, embed_dim)

        for layer in self.Layers:
            edge_features = torch.cat(
                [rigid.origin.unsqueeze(-1).dist(rigid.origin).unsqueeze(-1), rel_pos_embedding], dim=-1
            )
            # edge_features: (seq_len, seq_len, embed_dim)
            # edge_features = get_edge_features(rigid, rel_pos_embedding)
            node_features, rigid = layer(node_features, edge_features, rigid)

        # refine code.

        return node_features, rigid