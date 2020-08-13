from torch.nn.functional import mse_loss
from src.utils import gram_matrix


def criterion(content_features, style_features, output_features, content_weight=1, style_weight=1e5):
    # Content Loss
    content_loss = mse_loss(content_features[0], output_features[-1])
    
    # Style Loss
    style_loss = 0
    for s, o in zip(style_features, output_features):
        style_texture = gram_matrix(s)
        output_texture = gram_matrix(o)
        style_loss += mse_loss(style_texture, output_texture)
        
    # Total loss
    loss = content_weight * content_loss + style_weight * style_loss
    return loss