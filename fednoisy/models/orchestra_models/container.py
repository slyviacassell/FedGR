import torch.nn as nn
import torch

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder # DecoderContainer or nn.Module

    def forward(self, x, enable_encoder=True, enable_decoder=True, return_dict=False, full_heads=False):
        assert enable_encoder or enable_decoder, "At least one of the encoder or decoder should be enabled."
        if enable_encoder:
            x = self.encoder(x)
        if enable_decoder:
            x = self.decoder(x, return_dict=return_dict, full_heads=full_heads)
        return x

    def get_embedding(self, x):
        x = self.encoder(x)
        return x

    def get_model_list(self):
        return [self.encoder, self.decoder]
    

class DecoderContainer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.decoder_dict = nn.ModuleDict()

    def register_decoder(self, decoder, name=None):
        if name is None:
            name = type(decoder).__name__
        self.decoder_dict[name] = decoder

    def set_main_decoder(self, name):
        self.main_name = name

    def forward(self, x, return_dict=False, full_heads=False) -> dict:
        if full_heads:
            if return_dict:
                out = {}
                for name, decoder in self.decoder_dict.items():
                    out[name] = decoder(x)
            else:
                out = [decoder(x) for decoder in self.decoder_dict.values()]
        else:
            out = self.decoder_dict[self.main_name](x)
        return out
    
    def call_decoder(self, x, decoder_name) -> torch.Tensor:
        return self.decoder_dict[decoder_name](x)