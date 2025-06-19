import torch.nn as nn
from model import DAE_KAN_Attention
from histopathology_dataset import *

class IntermediateModel(nn.Module):
    def __init__(self, original_model):
        super(IntermediateModel, self).__init__()
        self.encoder1 = original_model.encoder1
        self.encoder2 = original_model.encoder2
        self.encoder3 = original_model.encoder3
        self.bottleneck_encoder1 = original_model.bottleneck_encoder1
        self.attn1 = original_model.attn1
        self.bottleneck_encoder2 = original_model.bottleneck_encoder2
        self.attn2 = original_model.attn2

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.bottleneck_encoder1(x)
        x = self.attn1(x)
        x = self.bottleneck_encoder2(x)
        x = self.attn2(x)
        return x

if __name__ == '__main__':
    # Instantiate the intermediate model with the loaded model
    model = DAE_KAN_Attention()
    checkpoint = torch.load("../histo-dae/go2j1m2y/checkpoints/epoch=3-step=9528.ckpt")

    intermediate_model = IntermediateModel(model)

    # Example input
    input_data = torch.randn(1, 3, 224, 224)

    # Get the output up to attn2
    output_data = intermediate_model(input_data)
    print(output_data.shape)
