import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision
from torchvision.transforms import Resize

class VisRes(nn.Module):
    def __init__(self):
        super(VisRes, self).__init__()
        # Load a pre-trained ResNet18
        self.pretrained = torchvision.models.resnet18(pretrained=True)
        # Modify the first layer to accept 2-channel input
        self.pretrained.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Remove the fully connected layer
        self.pretrained.fc = nn.Identity()

        # Determine the output size of the ResNet
        with torch.no_grad():
            dummy_data = torch.zeros(1, 2, 224, 224)
            output_size = self.pretrained(dummy_data).view(1, -1).size(1)
        print(output_size)
        # Add a new fully connected layer
        self.fc = nn.Linear(output_size, 9)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
if __name__=="__main__":
    a=VisRes()
