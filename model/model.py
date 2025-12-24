from pathlib import Path

import torch
import torch.nn as nn

"""
For classfisying images of MNIST dimensions
"""

class ResNet(nn.Module):
    class ResNetBlock(nn.Module):
        def __init__(self, in_size, out_size, depth, bias = False):
            """
            One 'skip' in the ResNet.

            Args:
                in_size
                out_size
                depth: number of hidden layers
                bias: learning an additive bias?

            Hard-coded hyperparameters:
                Non-linearity -- ReLU
                Normalization -- LayerNorm
                Hidden Dims -- out_size

            TODO Add in modularity for easy adjusting of hyperparameters
            """
            super().__init__()
            self.in_size = in_size
            self.out_size = out_size
            self.depth = depth
            self.bias = bias

            layers = []
            in_size_x = in_size
            for i in range(depth):
                layers.append(torch.nn.Linear(in_size_x, out_size,bias=bias))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.LayerNorm(out_size))
                in_size_x = out_size
            layers.append(torch.nn.Linear(in_size_x, out_size, bias=bias))

            self.block = torch.nn.Sequential(*layers)
        
            if in_size == out_size:
                self.residual = torch.nn.Identity()
            else:
                self.residual = torch.nn.Linear(in_size, out_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.residual(x) + self.block(x)
    
    def __init__(
            self, 
            h = 28, # MNIST default dimensions
            w = 28, # MNIST default dimensions
            out_classes = 2, # binary classification
            num_blocks = 2,
            depth_blocks = 2,
            hidden_dim = 128,
            bias = False
    ):
        """
        Simple Linear Residual Network for Classification.
        Currently flattens the image; plan to expand to ConvResNet in the future

        Args:
            h: height of image
            w: width of image
            out_classes: number of classes
            num_blocks: number of residual blocks for skipping
            depth_blocks: number of hidden layers in each residual block
            hidden_dim: dimensions of output of residual blocks
            bias: learning an additive bias?


        Hard-coded hyperparamters as defined in ResNetBlock.
        TODO Change shape to allow array for custom sizes of hidden blocks
        """
        super().__init__()
    
        self.h = h
        self.w = w
        self.color_channels = 1 # assumed grayscaled based on Tumor dataset definition
        self.out_channels = out_classes
        self.num_blocks = num_blocks
        self.depth_blocks = depth_blocks
        self.hidden_dim = hidden_dim
        self.bias = bias

        in_size_x = h * w * self.color_channels
        out_size = hidden_dim
        layers = []
        for i in range(num_blocks):
            layers.append(self.ResNetBlock(in_size_x, out_size, depth_blocks, bias=bias))
            in_size_x = out_size
        layers.append(torch.nn.Linear(in_size_x, out_classes, bias=bias))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.h * self.w * self.color_channels)
        # Ensure the input has a floating-point dtype to match model weights;
        # if input comes as a ByteTensor (e.g., raw image bytes), convert and normalize.
        if not x.dtype.is_floating_point:
            x = x.float() / 255.0
        return self.sequential(x)

