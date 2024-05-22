import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=50):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False), nn.BatchNorm2d(out_channels),  #it cancel the bias
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False), nn.BatchNorm2d(out_channels),  # it cancel the bias
            nn.ReLU(inplace=True)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x,t):
        return self.conv(x) + self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(t.shape[0], 1, device=device)], dim=-1)
        return embeddings
class AttentionBlock(nn.Module):
    def __init__(self, Fg,Fx, inter_channel):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(Fg, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(Fx,inter_channel, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(inter_channel)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channel,1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        psi = F.interpolate(psi, scale_factor=2, mode='bilinear', align_corners=True)
        return x*psi
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], attention=False, time_emb_dim=50):
        super(UNet, self).__init__()
        self.attention = attention
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        #trivial solution is to use nn.MaxPool2d(kernel_size=2, stride=2) but it will not work for odd size of the image
        #Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        #Up part of UNet
        for feature in features[::-1]:
            self.ups.append(AttentionBlock(Fg=2*feature, Fx=feature, inter_channel=feature*2)) if self.attention else nn.Identity()
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) #Upsample
            self.ups.append(DoubleConv(feature*2, feature))


        self.bottleneck = DoubleConv(features[-1],features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 3): #Due to the size of self.ups is 2 times the size of features (steps) each iteration will go thro 2 in the up li
            skip_connection = skip_connections[idx//3]
            attention = self.ups[idx](x, skip_connection) if self.attention else skip_connection
            x = self.ups[idx+1](x) #Up Sample   
            if x.shape != attention.shape:
                x = tf.Resize((skip_connection.shape[2], skip_connection.shape[3]))(x) #Torch vision resize -
            concat_skip = torch.cat((attention, x), dim=1) #add skip connection along channel dim
            x = self.ups[idx+2](concat_skip)
        return self.final_conv(x)

class UNet_With_T(nn.Module):
    #Create Defusion model , based on the UNet model with the addition of the SinusoidalPositionalEmbedding
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], attention=True, time_emb_dim=50):
        super(UNet_With_T, self).__init__()
        self.UNet = UNet(in_channels, out_channels, features, attention)
        self.time_emb = SinusoidalPositionalEmbedding(time_emb_dim)

    def forward(self, x, t):
        return self.UNet(torch.cat([x, self.time_emb(t)], dim=1))



def test():
    #Test the Defusion model
    x = torch.randn((1,3,64,64))
    model = Defusion(in_channels=3, out_channels=3, features=[64, 128, 256, 512], attention=True, time_emb_dim=50)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape #Check if the output shape is the same as the input shape
    print('Success')

if __name__ == '__main__':
    test()