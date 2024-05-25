import torch
# Rigid body structure

class Rigid:
    def __init__(self, origin, rot):
        self.origin = origin
        self.rot = rot
        self.shape = self.origin.shape

    def to(self, device):
        return Rigid(self.origin.to(device), self.rot.to(device))


def init_rigid(seq_len):
    '''create rigid body structure
    각 아미노산의 3차원 좌표, rotation matrix 생성
    '''
    origin = torch.zeros(seq_len, 3) # rigid origin, 3dim
    rot = torch.eye(3).unsqueeze(0)
    return Rigid(origin, rot) # 