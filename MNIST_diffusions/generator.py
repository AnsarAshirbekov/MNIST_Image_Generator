import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

T = 1000

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_hat = torch.cumprod(alphas, dim=0)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity

        return F.relu(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attention = torch.bmm(q.permute(0,2,1), k)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(v, attention.permute(0,2,1))
        out = out.view(B, C, H, W)

        return out + x

class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder
        self.down1 = ResBlock(2, 64)
        self.down2 = ResBlock(64, 128)
        self.attention = SelfAttention(128)

        #BotLeneck
        self.bottleneck = ResBlock(128, 128)

        #Decoder
        self.up1 = ResBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

        self.pool = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)

        self.label_embedding = nn.Embedding(10, 64)

    def forward(self, x, t, labels):

        t = t.float().view(-1,1,1,1) / T

        t_map = t.expand(-1,1,64,64)

        x = torch.cat([x,t_map], dim=1)

        x1 = self.down1(x)

        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(-1, 64, 1, 1)
        label_embed = label_embed.expand(
            -1,
            64,
            x1.shape[2],
            x1.shape[3]
        )

        x1 = x1 + label_embed

        x2 = self.pool(x1)

        x2 = self.down2(x2)

        x2 = self.pool2(x2)

        x3 = self.bottleneck(x2)
        x3 = self.attention(x3)

        x3 = F.interpolate(
            x3,
            size=(x1.shape[2], x1.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        x3 = self.up1(x3)

        x3 = x3 + x1

        out = self.final(x3)

        return out
        
    
model = NoisePredictor().to(device)

model.load_state_dict(
    torch.load("diffusion_mnist_model1.pth", map_location=device)
)

def generate(digit):

    model.eval()

    img = torch.randn(1, 1, 64, 64).to(device)

    with torch.no_grad():

        target_digit = torch.tensor([digit], device=device)

        for i in reversed(range(1, T)):

            t = torch.tensor([i], device=device)

            predicted_noise = model(img, t, target_digit)

            alpha = alphas[i]
            alpha_hat_t = alpha_hat[i]
            beta = betas[i]

            if i > 1:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (
                1 / torch.sqrt(alpha)
            ) * (
                img - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t))
                * predicted_noise
            ) + torch.sqrt(beta) * noise

    return img

digit = int(input("Введите цифру для рисования: "))
if digit < 0 or digit > 9:
    raise ValueError("Введите цифру от 0 до 9")

img = generate(digit)

img = img.clamp(0,1)

img = img.cpu().detach()[0][0]

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()