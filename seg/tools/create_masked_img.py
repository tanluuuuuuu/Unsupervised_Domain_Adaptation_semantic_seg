import numpy as np
import cv2
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode 
from PIL import Image

def generate_mask(imgs, mask_ratio, mask_block_size=64):
    H, W, C = imgs.shape

    input_mask = torch.rand(round(H / mask_block_size), round(W / mask_block_size))
    input_mask = (input_mask > mask_ratio).float()
    input_mask = input_mask.unsqueeze(0, ) 
    input_mask = resize(input_mask, size=(H, W), interpolation=InterpolationMode.NEAREST)
    input_mask = input_mask.permute(1, 2, 0)
    return input_mask

def mask_image(imgs, mask_ratio, mask_block_size):
    input_mask = generate_mask(imgs, mask_ratio, mask_block_size)
    return np.multiply(imgs, input_mask) 

if __name__ == "__main__":
    img_path = "/media/uulnat/New Volume1/Unsupervised_Domain_Adaptation_semantic_seg/src/MIC/seg/data/cityscapes/leftImg8bit/test/bielefeld/bielefeld_000000_000856_leftImg8bit.png"
    img = Image.open(img_path)
    img = np.array(img)

    img = torch.from_numpy(img)

    masked_img = mask_image(img, 0.7, 64).numpy()
    cv2.imwrite("masked.png", masked_img)