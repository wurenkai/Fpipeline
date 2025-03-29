import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed, sample_mask, sample_mask_all
from models.model import GPTConfig, GPT
import argparse
from tqdm import tqdm
from PIL import Image
import os
import time
from torch.nn import functional as F


class ICTEnsemble:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device(f"cuda:{opts.GPU_ids}" if torch.cuda.is_available() else "cpu")

        self.model_config = GPTConfig(
            512, opts.image_size * opts.image_size,
            embd_pdrop=0.0, resid_pdrop=0.0,
            attn_pdrop=0.0, n_layer=opts.n_layer,
            n_head=opts.n_head, n_embd=opts.n_embd,
            BERT=opts.BERT, use_gelu2=opts.GELU_2
        )

        self.model = GPT(self.model_config).to(self.device)
        checkpoint = torch.load(opts.ckpt_path)

        if opts.ckpt_path.endswith('.pt'):
            self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint['model'])

        self.C = torch.from_numpy(np.rint(127.5 * (np.load('kmeans_centers.npy') + 1.0)))

    def preprocess_image(self, image_path, mask_path):
        input_image = Image.open(image_path).convert("RGB")
        x = input_image.resize((self.opts.image_size, self.opts.image_size), resample=Image.BILINEAR)
        x = torch.from_numpy(np.array(x)).view(-1, 3).float()
        a = ((x[:, None, :] - self.C[None, :, :]) ** 2).sum(-1).argmin(1)

        input_mask = Image.open(mask_path).convert("L")
        y = input_mask.resize((self.opts.image_size, self.opts.image_size), resample=Image.NEAREST)
        y = torch.from_numpy(np.array(y) / 255.).view(-1)
        y = (y > 0.5).float()

        a_tensor = torch.stack([a] * self.opts.n_samples, dim=0)
        b_tensor = torch.stack([y] * self.opts.n_samples, dim=0)
        a_tensor *= (1 - b_tensor).long()

        return a_tensor, b_tensor

    def generate_predictions(self, a_tensor, b_tensor):
        self.model.eval()
        with torch.no_grad():
            if self.opts.sample_all:
                pixels = sample_mask_all(
                    self.model,
                    context=a_tensor,
                    length=self.opts.image_size * self.opts.image_size,
                    num_sample=self.opts.n_samples,
                    top_k=self.opts.top_k,
                    mask=b_tensor,
                    no_bar=self.opts.no_progressive_bar
                )
            else:
                pixels = sample_mask(
                    self.model,
                    context=a_tensor,
                    length=self.opts.image_size * self.opts.image_size,
                    num_sample=self.opts.n_samples,
                    top_k=self.opts.top_k,
                    mask=b_tensor,
                    no_bar=self.opts.no_progressive_bar
                )
        return pixels

    def ensemble_predictions(self, pixels):
        images = [self.C[p].view(self.opts.image_size, self.opts.image_size, 3)
                  for p in pixels]
        images = torch.stack(images, dim=0)

        ensemble_result = torch.median(images, dim=0)[0]
        return ensemble_result, images

    def save_results(self, images, ensemble_result, img_name):
        for i, img in enumerate(images):
            current_url = os.path.join(self.opts.save_url, f'condition_{i + 2}')
            os.makedirs(current_url, exist_ok=True)
            current_img = img.numpy().astype(np.uint8)
            tmp = Image.fromarray(current_img)
            tmp.save(os.path.join(current_url, img_name))

        ensemble_url = os.path.join(self.opts.save_url, f'condition_{1}')
        os.makedirs(ensemble_url, exist_ok=True)
        ensemble_img = ensemble_result.numpy().astype(np.uint8)
        ensemble_tmp = Image.fromarray(ensemble_img)
        ensemble_tmp.save(os.path.join(ensemble_url, img_name))

    def process_batch(self, img_list, mask_list):
        start_time = time.time()

        for x_name, y_name in zip(img_list, mask_list):
            if x_name != y_name:
                print("### Something Wrong ###")
                continue

            image_url = os.path.join(self.opts.image_url, x_name)
            mask_url = os.path.join(self.opts.mask_url, y_name)
            a_tensor, b_tensor = self.preprocess_image(image_url, mask_url)

            pixels = self.generate_predictions(a_tensor, b_tensor)

            ensemble_result, images = self.ensemble_predictions(pixels)

            img_name = f"{x_name[:-4]}.png"
            self.save_results(images, ensemble_result, img_name)

            print(f"Finish {img_name}")

        end_time = time.time()
        print(f"This test totally costs {end_time - start_time:.5f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--BERT', action='store_true', help='BERT model, Image Completion')
    parser.add_argument('--image_url', type=str, default='', help='the folder of image')
    parser.add_argument('--mask_url', type=str, default='', help='the folder of mask')
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=32, help='input sequence length: image_size*image_size')
    parser.add_argument('--n_layer', type=int, default=14)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--GELU_2', action='store_true', help='use the new activation function')
    parser.add_argument('--save_url', type=str, default='./', help='save the output results')
    parser.add_argument('--n_samples', type=int, default=8, help='sample cnt')
    parser.add_argument('--sample_all', action='store_true', help='sample all pixel together, ablation use')
    parser.add_argument('--skip_number', type=int, default=0,
                        help='since the inference is slow, skip the image which has been inferenced')
    parser.add_argument('--no_progressive_bar', action='store_true', help='')

    opts = parser.parse_args()

    ensemble = ICTEnsemble(opts)

    img_list = sorted(os.listdir(opts.image_url))
    mask_list = sorted(os.listdir(opts.mask_url))

    if opts.skip_number > 0:
        img_list = img_list[opts.skip_number - 1:]
        mask_list = mask_list[opts.skip_number - 1:]
        print(f"Resume from {img_list[0]}")

    if opts.BERT:
        ensemble.process_batch(img_list, mask_list)


if __name__ == '__main__':
    main()