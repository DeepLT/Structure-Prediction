import torch
import numpy as np
from constants import *

def get_one_hot(targets, nb_classes=20):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

def get_encoding(sequence):
    one_hot_amino = get_one_hot(np.array([res_to_num(x) for x in sequence]))
    return one_hot_amino # (seq_len, nb_classes=20)

# def encoding(sequence):
#     nb_classes = len(res_types)
#     res_to_num = np.array([res_to_num(x) for x in sequence])

#     res = np.eye(nb_classes)[np.array(res_to_num).reshape(-1)]
#     encoding = res.reshape(list(targets.shape) + [nb_classes])
#     return encoding

def get_rel_pos(sequence, rel_pos_dim): # node의 상대 위치 반환
    # broadcasting에 의해 tensor 크기 맞춰짐 
    # (1, seq_len) - (seq_len, 1) -> (seq_len, seq_len)
    rel_pos = torch.arange(len(sequence)).unsqueeze(0) - torch.arange(len(sequence)).unsqueeze(-1)

    # rel_pos_dim 의 두 배 만큼 clamp, 모든 값 양수 되도록 조정
    rel_pos = rel_pos.clamp(min=-rel_pos_dim, max=rel_pos_dim) + rel_pos_dim
    return rel_pos # rel_pos : (seq_len, seq_len), 값은 (0, 2*rel_pos_dim) 의 범위 가짐

def get_edge_features(rigid, rel_pos_embedding):
    # rigid.origin : rigid의 좌표
    # rigid.origin.unsqueeze(-1) : (seq_len, 3)
    # dist : 유클리드 거리 계산, 브로드캐스팅 발생 (seq_len, seq_len)
    # (seq_len, seq_len, embed_dim)
    return torch.cat([rigid.origin.unsqueeze(-1).dist(rigid.origin).unsqueeze(-1), rel_pos_embedding], dim=-1)