import torch
import torch.nn as nn
import torchvision.transforms.functional as F
#import random
#from PIL import Image

#import cv2
#import numpy as np
import pytensor.tensor as pt
from pytensor import function



class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024)
        )

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            self.upconv_block(1024, 512),
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        enc5 = self.encoder[4](enc4)

        # Decoder
        dec1 = self.decoder[0](enc5)
        dec2 = self.decoder[1](dec1 + enc4)
        dec3 = self.decoder[2](dec2 + enc3)
        dec4 = self.decoder[3](dec3 + enc2)
        out = self.decoder[4](dec4 + enc1)

        return out


def predict(masked_image):
    path = "D:/hackpsuS25/hackpsu_S25/unet.pth"
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval() 

    masked_image = masked_image.unsqueeze(0)  # Add batch dimension
    masked_image = masked_image.to("cpu")

    with torch.no_grad():
            generated_imgs = model(masked_image)
    def denorm(img):
        return (img * 0.5) + 0.5
    idx = 0
    return denorm(generated_imgs[idx].cpu().permute(1, 2, 0))

""""
def add_random_mask(image_path, mask_ratio=0.5, mask_color=(0, 0, 0)):

    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)

    height, width = image_tensor.shape[1], image_tensor.shape[2]
    mask_area = int(height * width * mask_ratio)

    mask_width = random.randint(1, width)
    mask_height = mask_area // mask_width
    
    if mask_height > height:
        mask_height = height
        mask_width = mask_area // mask_height
    
    x_start = random.randint(0, width - mask_width) if width > mask_width else 0
    y_start = random.randint(0, height - mask_height) if height > mask_height else 0

    mask = torch.ones_like(image_tensor[:, y_start:y_start+mask_height, x_start:x_start+mask_width])
    mask[0, :, :] *= mask_color[0] / 255.0
    mask[1, :, :] *= mask_color[1] / 255.0
    mask[2, :, :] *= mask_color[2] / 255.0

    image_tensor[:, y_start:y_start+mask_height, x_start:x_start+mask_width] = mask
    return image_tensor


image_path = "D:/hackpsuS25/hackpsu_S25/face/output_image_0.jpg"
masked_image = add_random_mask(image_path, mask_ratio=0.1, mask_color=(0, 0, 0))



i = cv2.cvtColor(masked_image.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)

cv2.imshow('Image', i)

cv2.waitKey(0)  # Wait for a key press to close the window


predict(masked_image)

i = cv2.cvtColor(predict(masked_image).numpy(), cv2.COLOR_BGR2RGB)

cv2.imshow('Image', i)

cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()"""