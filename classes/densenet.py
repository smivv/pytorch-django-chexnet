import torch
import torchvision


class DenseNet121(torch.nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, out_size),
            torch.nn.Sigmoid()
        )
        # self.densenet121.classifier = torch.backends.cudnn.convert(self.densenet121.classifier, torch.nn)

    def forward(self, x):
        x = self.densenet121(x)
        return x
