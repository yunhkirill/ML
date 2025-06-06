import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolutional_encoder import ConvolutionalEncoder


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (dict): Contains 'images', 'n_way', 'n_support', 'n_query'
        Returns:
            torch.Tensor: loss value
            dict: Contains 'loss', 'acc', and 'y_hat' (predictions)
        """
        sample_images = sample['images'].to(self.device)
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support = sample_images[:, :n_support]
        x_query = sample_images[:, n_support:]

        x_support = x_support.contiguous().view(n_way * n_support, *x_support.size()[2:])
        x_query = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])

        z_support = self.encoder(x_support)
        z_query = self.encoder(x_query)

        z_support = z_support.view(n_way, n_support, -1)
        z_proto = z_support.mean(1)

        dists = torch.cdist(z_query, z_proto)

        scores = -dists

        y_query = torch.arange(0, n_way).view(n_way, 1).repeat(1, n_query).view(-1).to(self.device)

        loss_val = F.cross_entropy(scores, y_query)

        _, y_hat = scores.max(1)

        acc_val = torch.eq(y_hat, y_query).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
        }


def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (Class ProtoNet)
    """
    encoder = ConvolutionalEncoder()

    return ProtoNet(encoder)