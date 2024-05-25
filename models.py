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

    def forward(self, node_features, edge_features, rigid, mask):
        return self.IPA(
            node_features,
            edge_features,
            rotations = rigid.rot,
            translations = rigid.origin,
            mask = mask
        )
    
class RigidUpdate(nn.Module):
    # node_feature 기반으로 rigid update
    def __init__(self, node_dim):
        super().__init__()

        # origin, rot 나타내는 6개 차원으로 linear
        self.linear = nn.Linear(node_dim, 6)

    def forward(self, node_features, mask):
        # linear 수행 후 rot(quaternion), origin 으로 분리
        rot, origin = self.linear(node_features).chunk(2, dim=-1)

        # 업데이트 마스크 적용
        if mask is not None:
            rot = mask[:, None] * rot
            t = mask[:, None] * t

        # 정규화
        norm = (1 + rot.pow(2).sum(-1, keepdim=True)).pow(1 / 2)
        b, c, d = (rot / norm).chunk(3, dim=-1)
        a = 1 / norm
        a, b, c, d = a.squeeze(-1), b.squeeze(-1), c.squeeze(-1), d.squeeze(-1)
        rot = F.quaternion_to_rotation_matrix(torch.cat([a, b, c, d], dim=-1))

        return Rigid(origin, rot)


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