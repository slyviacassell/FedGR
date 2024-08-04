import torch
import torch.nn.functional as TF
import numpy as np

from .hook import SerialClientTrainerHook, SerialServerTrainerHook

class KNNClassificationGlboalHook(SerialClientTrainerHook):
    def __init__(self, k) -> None:
        super().__init__()
        