import torch
import torch.nn.functional as TF
import numpy as np

from .hook import SerialClientTrainerHook

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
)

class LocalNoisePrior(SerialClientTrainerHook):
    def __init__(self) -> None:
        super().__init__()

    def on_init(self, client_trainer, *args, **kwargs):
        client_trainer.local_noise_priors = [None] * client_trainer.num_clients

    def on_client_training_start(self, client_trainer, *args, **kwargs):
        if client_trainer.local_noise_priors[client_trainer.l_cid] is None:
            client_trainer.local_noise_priors[client_trainer.l_cid] = self.get_noise_prior(client_trainer)

    def get_noise_prior(self, client_trainer):
        train_dataloader = client_trainer.dataset.get_dataloader(client_trainer.g_cid, train=True, batch_size=client_trainer.batch_size)

        # get the label distribution of the data loader
        noisy_labels = []
        for batch in train_dataloader:
            noisy_labels.append(batch["noisy_label"])
        noisy_labels = torch.cat(noisy_labels).numpy()
        label_dist = np.bincount(noisy_labels, minlength=CLASS_NUM[client_trainer.args.dataset])
        label_dist = label_dist / label_dist.sum()

        return label_dist
            
