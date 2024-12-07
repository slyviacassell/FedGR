import torch.nn as nn
import torchvision

from .container import EncoderDecoder, DecoderContainer
from .head import (
    OrchestraHead, 
    ClassificationHead, 
    LinearHead,
    SimSiamHead,
    BYOLHead,
    SimSiamPredictionHead,
)
from .resnet import ResNet18, ResNet34

def build_model(model_name, ssl_method: str, num_classes: int=10, rep_dim: int=512, dataset: str="CIFAR10"):
    if model_name == "ResNet18":
        encoder = ResNet18()
    elif model_name == "ResNet34":
        encoder = ResNet34()
    elif model_name == "ResNet50":
        encoder = torchvision.models.resnet50(pretrained=True)
        encoder.fc = nn.Linear(2048, 2048)
        encoder.out_features = 2048
    else:
        raise ValueError(
            f"Unrecognized model: {model_name}. Currently only support 'ResNet18'."
        )
    
    decoder = DecoderContainer()
    register_cls_head(decoder, encoder.out_features, num_classes, hidden_features=rep_dim)
    register_ssl_heads(ssl_method, decoder, encoder.out_features, rep_dim)
    
    model = EncoderDecoder(encoder, decoder)
    
    return model

def register_ssl_heads(ssl_method, decoder, feat_dim, rep_dim):
    if ssl_method == "orchestra":
        register_orchestra_head(decoder, feat_dim, rep_dim)
    elif ssl_method == "simsiam":
        register_simsiam_head(decoder, feat_dim, rep_dim)
    elif ssl_method == "byol":
        register_byol_head(decoder, feat_dim, rep_dim)
    elif ssl_method == "simplessl":
        register_simplessl_head(decoder, feat_dim, rep_dim)
    elif ssl_method == "fedprox_like":
        register_fedprox_like_head(decoder, feat_dim, rep_dim)
    else:
        raise ValueError(
            f"Unrecognized ssl_method: {ssl_method}"
        )
    
def register_fedprox_like_head(decoder, feat_dim: int, rep_dim: int):
    simplessl_head = SimSiamPredictionHead(feat_dim, out_dim=rep_dim)
    decoder.register_decoder(simplessl_head, "simplessl_head")

def register_simplessl_head(decoder, feat_dim: int, simplessl_dim: int):
    # simplessl_head = OrchestraHead(feat_dim, simplessl_dim)
    simplessl_head = SimSiamPredictionHead(feat_dim, out_dim=simplessl_dim)
    decoder.register_decoder(simplessl_head, "simplessl_head")

def register_orchestra_head(decoder, feat_dim: int, orchestra_dim: int):
    orchestra_head = OrchestraHead(feat_dim, orchestra_dim)
    decoder.register_decoder(orchestra_head, "orchestra_head")

def register_simsiam_head(decoder, feat_dim: int, proj_dim: int=512):
    simsiam_head = SimSiamHead(in_dim=feat_dim, out_dim=proj_dim)
    decoder.register_decoder(simsiam_head, "projector_predictor")

def register_byol_head(decoder, feat_dim: int, proj_dim: int=512):
    byol_head = BYOLHead(feat_dim=feat_dim, projection_size=proj_dim, prediction_size=proj_dim, hidden_size=2048)
    decoder.register_decoder(byol_head, "projector_predictor")

def register_cls_head(decoder, feat_dim: int, num_classes: int, hidden_features=None):
    cls_head = ClassificationHead(feat_dim, num_classes, hidden_features=hidden_features)
    decoder.register_decoder(cls_head, "cls_head")
    decoder.set_main_decoder("cls_head")
 