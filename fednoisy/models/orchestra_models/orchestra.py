import torch.nn as nn
import torch.nn.functional as TF
import torchvision
import torch


class Orchestra(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, queue_size: int) -> None:
        super().__init__()
        self.backbone = backbone

        self.feat_dim = feat_dim
        self.K = queue_size
        self.register_buffer("queue", torch.randn(self.feat_dim, self.K))
        self.queue = TF.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        out = self.backbone(x)
        with torch.no_grad():
            if "orchestra_head" in out:
                keys = out["orchestra_head"]
                keys = TF.normalize(keys.detach().clone(), dim=1)
                self._dequeue_and_enqueue(keys)
        return out