import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_class: int, hidden_features=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_class = num_class

        if hidden_features is None:
            hidden_features = in_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
        )
        self.cls_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, num_class)
        )

    def forward(self, x, return_dict=False, *args, **kwargs):
        hidden = self.mlp(x)
        out = self.cls_head(hidden)
        if return_dict:
            return {"cls_logits": out, "cls_embedding": hidden}
        return out
        

class OrchestraHead(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=128):
        super(OrchestraHead, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features, bias=False),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x, *args, **kwargs):
        return self.linear(x)
    

class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return self.linear(x)
    

class SimSiamProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(SimSiamProjectionHead, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x, *args, **kwargs):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x
    

class SimSiamPredictionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=512):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h's input and output (z and p) is d = 2048, 
        and h's hidden layer's dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x, *args, **kwargs):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    

class SimSiamHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(SimSiamHead, self).__init__()
        self.predictor = SimSiamPredictionHead(in_dim, hidden_dim, out_dim)
        self.projector = SimSiamProjectionHead(in_dim, hidden_dim, out_dim)

    def forward(self, x, *args, **kwargs):
        z = self.projector(x)
        p = self.predictor(z)
        return {
            "z": z,
            "p": p,
        }
    

class BYOLProjPredHead(nn.Module):
    def __init__(self, dim, projpred_size, hidden_size=4096):
        super(BYOLProjPredHead,self).__init__()
        self.in_features = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projpred_size),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)
    

class BYOLHead(nn.Module):
    def __init__(self, feat_dim, projection_size, prediction_size, hidden_size=4096):
        super(BYOLHead,self).__init__()
        self.proj = BYOLProjPredHead(feat_dim, projection_size, hidden_size=hidden_size)
        self.pred = BYOLProjPredHead(projection_size, prediction_size, hidden_size=hidden_size)

    def forward(self, x, *args, **kwargs):
        z = self.proj(x)
        p = self.pred(z)
        return {
            "z": z,
            "p": p,
        }
        