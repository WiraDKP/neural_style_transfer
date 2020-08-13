from torch import nn
from torchvision.models import vgg19_bn


class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        
        model = vgg19_bn(pretrained=True).eval()
        self.model = model.features
        self.freeze()
        
    def forward(self, x, layers):
        features = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if i in layers:
                features.append(x)
        return features
    
    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False
