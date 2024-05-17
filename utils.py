import numpy as np
from constants import *

def get_one_hot(targets, nb_classes=21):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])

def get_encoding(sequence):
    one_hot_amino = get_one_hot(np.array([res_to_num(x) for x in sequence]))
    return one_hot_amino

# def encoding(sequence):
#     nb_classes = len(res_types)
#     res_to_num = np.array([res_to_num(x) for x in sequence])

#     res = np.eye(nb_classes)[np.array(res_to_num).reshape(-1)]
#     encoding = res.reshape(list(targets.shape) + [nb_classes])
#     return encoding