import torch
import torch.nn as nn
import torch.nn.functional as F
from rigid import *

class IPA(nn.Module):
    # Invariant Point Attention
    def __init__(self, node_dim, embed_dim):
        self.node_dim = node_dim
        self.embed_dim = embed_dim

    def forward(node_features, edge_features, rigid):
        return
    
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
    '''
    node_dim : amino acid 종류 20가지 차원
    hidden_dim : embedding 차원
    '''
    def __init__(self, node_dim, embed_dim, dropout_ratio):
        super().__init__()
        self.IPA = IPA(node_dim, embed_dim)
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
        out = self.IPA(node_features, edge_features, rigid)
        out = self.norm1(out)
        out = self.residual(out)
        out = self.norm2(out)

        # update rigid backbone

        return out


class StructureModule(nn.Module):
    # def __init__(self):
    
    # def forward(self):