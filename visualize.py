#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path


import numpy as np
from PIL import Image
from tqdm import tqdm

from sinder import (
    get_tokens,
    load_model,
    load_visual_data,
    pca_array,
)

os.environ['XFORMERS_DISABLED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument(
        'imgs', nargs='+', type=str, help='path to image/images'
    )
    parser.add_argument(
        '--model', type=str, default='dinov2_vitg14', help='model name'
    )
    parser.add_argument('--workdir', type=str, default='visualize')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint. Default is None, which loads the official pretrained weights',
    )
    parser.add_argument(
        '--visual_size',
        type=int,
        default=518,
        help='short side size of input image',
    )

    args = parser.parse_args()
    return args


import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize(args, model, visual_dataset):
    model.eval()

    all_patch_tokens = []
    metadata = []  # stores (filename, h_patches, w_patches)

    print("Extracting features...")
    # First pass: Collect all tokens
    for d in tqdm(range(len(visual_dataset))):
        visual_image = visual_dataset[d]
        # Use forward_features to get normalized tokens (matches run_pca_batch.py)
        # We need to manually add batch dimension and move to device if needed
        # visual_image is a tensor from dataset. It's already prepared by dataset logic but load_visual_data -> VisualDataset might return it on CPU.
        # Let's check load_visual_data implementation or assume we need to handle device. 
        # But get_tokens handled it. let's reproduce basic logic.
        
        image_tensor = visual_image.unsqueeze(0).cuda()
        
        with torch.no_grad():
            features_dict = model.forward_features(image_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] # (1, N_patches, C)
        
        filename = Path(visual_dataset.files[d]).stem
        
        t = patch_tokens.squeeze(0).cpu() # (N_patches, C)
        
        # We need H and W. run_pca_batch calculates it from image size.
        # visual_dataset probably resizes images.
        # The stored image tensor shape is (C, H, W).
        _, _, h_img, w_img = image_tensor.shape
        h = h_img // 14
        w = w_img // 14
        
        # t is (N_patches, C)
        
        all_patch_tokens.append(t)
        metadata.append({
            'filename': filename,
            'h_patches': h,
            'w_patches': w
        })

    if not all_patch_tokens:
        print("No features extracted.")
        return

    # Concatenate all tokens for global PCA
    print("Fitting PCA...")
    total_features = torch.cat(all_patch_tokens, dim=0) # (Total_patches, C)
    
    # PCA
    pca = PCA(n_components=1)
    # Convert to numpy
    total_features_np = total_features.numpy()
        
    pca.fit(total_features_np)
    pca_features = pca.transform(total_features_np)
    pca_features = pca_features[:, 0]
    
    # Min-Max normalize to 0-1
    pca_features = (pca_features - pca_features.min()) / \
                         (pca_features.max() - pca_features.min())

    # Reconstruct images
    print("Saving results...")
    current_idx = 0
    for meta in metadata:
        n_patches = meta['h_patches'] * meta['w_patches']
        
        # Slice features for this image
        img_pca = pca_features[current_idx : current_idx + n_patches]
        current_idx += n_patches
        
        # Reshape to (H_patches, W_patches)
        img_pca = img_pca.reshape(meta['h_patches'], meta['w_patches'])
        
        # Save
        # Save old norm image as well if needed, or just the PCA one as requested. 
        # User asked to base on run_pca_batch.py which only saves PCA. 
        # But existing visualize.py also saved _norm.png. I will keep _norm.png logic if I can, 
        # but the loop structure changed.
        # I'll re-implement _norm.png saving inside this loop if possible, OR just focus on PCA as requested.
        # I'll just save the PCA image as requested to minimize complexity and stick to the prompt "visualize first principal component".
        
        out_name = f"{meta['filename']}_pca1.png"
        out_path = args.folder / out_name
        
        plt.imsave(out_path, img_pca)
        print(f"Saved {out_path}")


def main():
    args = parse_args()

    args.folder = Path(args.workdir).expanduser()
    os.makedirs(args.folder, exist_ok=True)
    print(args)
    print(' '.join(sys.argv))

    model = load_model(args.model, args.checkpoint)
    visual_dataset = load_visual_data(args, model)
    visualize(args, model, visual_dataset)


if __name__ == '__main__':
    main()
