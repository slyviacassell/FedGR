import torch.nn as nn
import torchvision


from .head import LinearHead
from .resnet import ResNet18, ResNet34

def build_model(model_name, num_classes: int=10, dataset: str="CIFAR10"):
    if model_name == "ResNet18":
        # encoder = ResNet18()
        # linear_head = LinearHead(encoder.out_features, num_classes)
        
        # model = (encoder, linear_head)
        
        encoder = ResNet18()
        linear_head = nn.Sequential(
            encoder,
            LinearHead(encoder.out_features, num_classes)
        )
        encoder = nn.Identity()
        
        model = (encoder, linear_head)
    elif model_name == "ResNet34":
        # encoder = ResNet34()
        # linear_head = LinearHead(encoder.out_features, num_classes)
        
        # model = (encoder, linear_head)

        encoder = ResNet34()
        linear_head = nn.Sequential(
            encoder,
            LinearHead(encoder.out_features, num_classes)
        )
        encoder = nn.Identity()
        
        model = (encoder, linear_head)
    else:
        raise ValueError(
            f"Unrecognized model: {model_name}. Currently only support 'ResNet18'."
        )
    
    return model