import torch
import torch.nn as nn
import torch.nn.functional as F
from ...components.attention_mechanisms.bam import BAM
from ...components.attention_mechanisms.eca import ECALayer
from ...components.kan.kan_layer import KANLayer
from .KANConv import KAN_Convolutional_Layer as KANCL

class Autoencoder_Encoder(nn.Module):

    def __init__(self, device: str = 'cpu', config=None):
        super(Autoencoder_Encoder, self).__init__()
        #Encoders
        
        self.config = config or {"use_kan": True, "kan_options": {}}
        
        if self.config.get("use_kan", True):
            self.kan = KANCL(
                n_convs=1,
                kernel_size=tuple(self.config["kan_options"].get("kernel_size", [5,5])),
                padding=tuple(self.config["kan_options"].get("padding", [2,2])),
                device=device
            )
        else:
            self.kan = nn.Identity()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),

        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True)
        )

        self.ECA_Net = ECALayer(64) if self.config.get("use_eca", True) else nn.Identity()

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True), 
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 384, kernel_size=3, padding=1, stride=2, output_padding=1), 
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True), 
        )

    def forward(self, x: torch.Tensor):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)


        residual1 = self.encoder1(x)
        residual2 = self.encoder2(residual1)

        out = self.kan(out)
        out = self.ECA_Net(out)
        out = self.decoder1(out) + residual2
        out = self.decoder2(out) + residual1
        return out, residual1, residual2


class Autoencoder_Decoder(nn.Module):

    def __init__(self, device: str = 'cuda'):
        super(Autoencoder_Decoder, self).__init__()
        #Encoders
        
        self.kan = KANCL(
                    n_convs = 1,
                    kernel_size = (5, 5),
                    padding = (2, 2),
                    device=device
                )

        self.encoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 384, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),

        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True)
        )

        self.ECA_Net = ECALayer(64)

        self.decoder1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True), 
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 384, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True), 
        )

        self.Output_Layer = nn.Sequential(
            nn.ConvTranspose2d(384, 3 , kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ELU(inplace=True), 
        )

        self.reconstructtion = KANCL(
                    n_convs = 1,
                    kernel_size = (3, 3),
                    padding = (1, 1),
                    device=device
                )

    def forward(self, x: torch.Tensor, residualEnc1, residualEnc2):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)


        out = self.kan(out)
        out = self.ECA_Net(out)
        residual1 = self.encoder1(x)
        residual2 = self.encoder2(residual1)

        out = self.decoder1(out) + residual2 + residualEnc2
        out = self.decoder2(out) + residual1 + residualEnc1
        out = self.Output_Layer(out)
        out = self.reconstructtion(out)


        return out

class Autoencoder_BottleNeck(nn.Module):

    def __init__(self, config=None): 
        self.config = config or {"use_bam": True}
        super(Autoencoder_BottleNeck, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ELU(inplace=True),
        )

        self.attn1 = BAM(384) if self.config.get("use_bam", True) else nn.Identity()

        self.encoder2 = nn.Sequential(
            nn.Conv2d(384, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
        )

        self.attn2 = BAM(16) if self.config.get("use_bam", True) else nn.Identity()

        # The VAE Code

        self.decoder = nn.Sequential(
           nn.ConvTranspose2d(16, 128, kernel_size=3, padding=1, stride=2, output_padding=1),
           nn.BatchNorm2d(128),
           nn.ELU(inplace=True), 
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder1(x)
        x = self.attn1(x)
        x = self.encoder2(x)
        z = self.attn2(x)
        x = self.decoder(x)

        return x, z

class DAE_KAN_Attention(nn.Module):
    def __init__(self, device: str = 'cuda', config=None):
        super().__init__()
        self.config = config or {}
        self.ae_encoder = Autoencoder_Encoder(device=device, config=self.config)
        self.bottleneck = Autoencoder_BottleNeck(config=self.config)
        self.ae_decoder = Autoencoder_Decoder(device=device, config=self.config)

    def forward(self, x):

        encoded, residual1, residual2 = self.ae_encoder(x)
        decoded, z =  self.bottleneck(encoded)
        decoded = self.ae_decoder(decoded, residual1, residual2)

        return encoded, decoded, z

class KAN_feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.KAN1 = KANLayer(16**3, 4096)
        self.KAN2 = KANLayer(4096, 2048)

    def forward(self, x):
        x = self.flat(x)
        x = self.KAN1(x)
        x = self.KAN2(x)

        return x
        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    from torchsummary import summary
    x_d = torch.randn(2, 128, 128, 128)
    x_e = torch.randn(10, 3, 128, 128).to('cuda')
    x_b = torch.randn(2, 384, 256, 256)
    model_en = Autoencoder_Encoder(device='cuda').to('cuda')
    model_de = Autoencoder_Decoder(device='cuda').to('cuda')
    model_bn = Autoencoder_BottleNeck().to('cuda')
    complete = DAE_KAN_Attention().to('cuda')
    x, y, z = complete(x_e)

    summary(complete, x_e)
    print(x.shape)
    # flat = nn.Flatten().to('cuda')
    # flatten = flat(z)
    #print(flatten.shape)
    #kan_feat = KANLayer(4096, 2048, device='cuda')

    #print(kan_feat(z).shape)
    #fe = KAN_feature_extractor().to('cuda')
    #summary(fe, z)
    #print(z)
    #print(y[1].shape)


    #out = model_de(x)
    #print(z)
    #summary(model_bn, x_b)

