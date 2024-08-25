import torch.nn as nn
import torchvision

from .container import EncoderDecoder, DecoderContainer
from .head import LinearHead
from .resnet import ResNet18, ResNet34

def build_model(model_name, num_classes: int=10, orchestra_dim: int=128, dataset: str="CIFAR10"):
    if model_name == "ResNet18":
        encoder = ResNet18()
        decoder = DecoderContainer()
        
        linear_head = LinearHead(encoder.out_features, num_classes)
        orchestra_head = LinearHead(encoder.out_features, orchestra_dim)
        decoder.register_decoder(linear_head, "linear_head")
        decoder.register_decoder(orchestra_head, "orchestra_head")
        decoder.set_main_decoder("linear_head")
        
        model = EncoderDecoder(encoder, decoder)
    else:
        raise ValueError(
            f"Unrecognized model: {model_name}. Currently only support 'ResNet18'."
        )
    
    return model