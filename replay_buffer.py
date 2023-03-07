import torch


class ReplayBuffer():
    def __init__(self):
        self.buffer = torch.tensor([])

    def cat(self, hist):
        self.buffer = torch.cat([self.buffer, hist.unsqueeze(0)])
