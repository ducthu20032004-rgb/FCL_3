#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_cifar10_npy.py
Visualizations & diagnostics for CIFAR-10 datasets stored as per-class .npy files.
Saves figures into a target folder (default: ./figures).

Usage (from repo root after building dataset/cifar10-classes/*.npy):
    python visualize_cifar10_npy.py \
        --data_dir dataset/cifar10-classes \
        --fig_dir figures \
        --grid_per_class 10 \
        --tsne_samples 2000 \
        --use_pretrained \
        --augmentations

If you are offline or don't want to download torchvision weights for ResNet18, omit --use_pretrained
and the t-SNE will compute on PCA-reduced raw pixels instead.

Requires: numpy, matplotlib, pillow
Optional (for t-SNE/PCA & augmentations): scikit-learn, torch, torchvision
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Optional deps
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from torchvision import transforms
    from torchvision.models import resnet18
    from torchvision.models import ResNet18_Weights
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# CIFAR-10 label names
CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_uint8_hwc(imgs: np.ndarray) -> np.ndarray:
    """
    Standardize a batch of images to uint8 HWC (N, H, W, 3) in [0,255].
    Accepts (N, 3, H, W) or (N, H, W, 3), float [0,1] or [0,255], or uint8.
    """
    assert imgs.ndim in (3,4), f"Unexpected ndim {imgs.ndim}"
    if imgs.ndim == 3:
        if imgs.shape[-1] == 3:
            imgs = imgs[None, ...]
        elif imgs.shape[0] == 3:
            imgs = np.transpose(imgs, (1,2,0))[None, ...]
        else:
            raise ValueError(f"Cannot infer channel position for shape {imgs.shape}")
    else:
        if imgs.shape[-1] == 3:
            pass
        elif imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0,2,3,1))
        else:
            raise ValueError(f"Cannot infer channel position for shape {imgs.shape}")
    if imgs.dtype == np.uint8:
        return imgs
    imgs = imgs.astype(np.float32)
    vmin, vmax = imgs.min(), imgs.max()
    if vmax <= 1.0 and vmin >= 0.0:
        imgs = imgs * 255.0
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    return imgs

def load_cifar10_npy_dir(data_dir: Path) -> Dict[int, np.ndarray]:
    """
    Load all .npy files in the folder, assume each file is a single class.
    Returns dict: {class_idx: np.ndarray of images (N,H,W,3) uint8}
    Infers class_idx from filename prefix (e.g., '0.npy', '7_frog.npy').
    """
    files = sorted([p for p in data_dir.glob("*.npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    per_class = {}
    for f in files:
        name = f.stem
        parts = name.split("_")
        try:
            cls = int(parts[0])
        except Exception:
            cls = len(per_class)
        arr = np.load(f, allow_pickle=True)
        imgs = to_uint8_hwc(arr)
        per_class[cls] = imgs
    return per_class

def stats_channel_mean_std(imgs_hwc_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = imgs_hwc_uint8.astype(np.float32) / 255.0
    mean = x.mean(axis=(0,1,2))
    std = x.std(axis=(0,1,2))
    return mean, std

def plot_bar(values: List[int], labels: List[str], title: str, save_path: Path, rotate=45):
    plt.figure(figsize=(10, 5))
    xs = np.arange(len(values))
    plt.bar(xs, values)
    plt.xticks(xs, labels, rotation=rotate, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def make_grid(images: List[np.ndarray], nrow: int, padding: int=2) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("No images for grid.")
    H, W, C = images[0].shape
    ncol = int(math.ceil(len(images) / nrow))
    grid = np.ones((nrow * H + (nrow - 1) * padding, ncol * W + (ncol - 1) * padding, C), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        r = idx // ncol
        c = idx % ncol
        rs = r * (H + padding)
        cs = c * (W + padding)
        grid[rs:rs+H, cs:cs+W, :] = img
    return grid

def save_image(arr_uint8_hwc: np.ndarray, path: Path):
    Image.fromarray(arr_uint8_hwc).save(path)

def compute_features(
    per_class: Dict[int, np.ndarray],
    max_samples: int,
    use_pretrained: bool
) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    rng = np.random.default_rng(42)
    classes_sorted = sorted(per_class.items())
    per_cls_cap = max_samples // len(classes_sorted) if max_samples else None
    for cls, imgs in classes_sorted:
        n = imgs.shape[0]
        take = min(n, per_cls_cap) if per_cls_cap else n
        idxs = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        for i in idxs:
            images.append(imgs[i])
            labels.append(cls)
    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int64)

    if use_pretrained and TORCH_AVAILABLE:
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            preprocess = weights.transforms()
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            with torch.no_grad():
                feats = []
                bs = 128
                for i in range(0, len(images), bs):
                    batch = images[i:i+bs]
                    pil = [Image.fromarray(im) for im in batch]
                    t = torch.stack([preprocess(p) for p in pil], dim=0)
                    out = model(t).cpu().numpy()
                    feats.append(out)
                features = np.concatenate(feats, axis=0)
        except Exception as e:
            print(f"[WARN] Pretrained feature extraction failed: {e}")
            print("Falling back to PCA on raw pixels.")
            use_pretrained = False
    if not use_pretrained:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available; cannot compute PCA/TSNE. Install scikit-learn or use --use_pretrained with torchvision.")
        X = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
        pca = PCA(n_components=min(50, X.shape[1]))
        features = pca.fit_transform(X)
    return features, labels

def save_tsne_scatter(features: np.ndarray, labels: np.ndarray, out_path: Path, title: str):
    if not SKLEARN_AVAILABLE:
        print("[WARN] scikit-learn not available; skipping t-SNE plot.")
        return
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, max_iter=1000, verbose=0)
    Y = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for cls in sorted(np.unique(labels)):
        idx = labels == cls
        plt.scatter(Y[idx,0], Y[idx,1], s=8, label=CIFAR10_LABELS.get(int(cls), str(int(cls))))
    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

def save_prototypes_tsne(per_class: Dict[int, np.ndarray], fig_path: Path, use_pretrained: bool):
    if use_pretrained and TORCH_AVAILABLE:
        try:
            weights = ResNet18_Weights.IMAGENET1K_V1
            preprocess = weights.transforms()
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            feats = []
            labels = []
            with torch.no_grad():
                for cls, imgs in sorted(per_class.items()):
                    take = min(500, imgs.shape[0])
                    pil = [Image.fromarray(imgs[i]) for i in range(take)]
                    t = torch.stack([preprocess(p) for p in pil], dim=0)
                    out = model(t).cpu().numpy()
                    feats.append(out.mean(axis=0, keepdims=True))
                    labels.append(cls)
            features = np.concatenate(feats, axis=0)
            labels = np.array(labels)
        except Exception as e:
            print(f"[WARN] Pretrained prototype extraction failed: {e}")
            return
    else:
        if not SKLEARN_AVAILABLE:
            print("[WARN] scikit-learn not available; skipping prototypes t-SNE.")
            return
        feats = []
        labels = []
        for cls, imgs in sorted(per_class.items()):
            X = imgs.reshape(imgs.shape[0], -1).astype(np.float32) / 255.0
            feats.append(X.mean(axis=0, keepdims=True))
            labels.append(cls)
        X = np.concatenate(feats, axis=0)
        pca = PCA(n_components=min(50, X.shape[1]))
        features = pca.fit_transform(X)
        labels = np.array(labels)

    if not SKLEARN_AVAILABLE:
        print("[WARN] scikit-learn not available; skipping prototypes t-SNE plot.")
        return
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=5, n_iter=1000, verbose=0)
    Y = tsne.fit_transform(features)
    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(labels):
        plt.scatter(Y[i,0], Y[i,1], s=80)
        plt.text(Y[i,0], Y[i,1], CIFAR10_LABELS.get(int(cls), str(int(cls))), fontsize=9)
    plt.title("Class prototypes (t-SNE)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=250)
    plt.close()

def save_random_grid(per_class: Dict[int, np.ndarray], fig_path: Path, per_class_n: int, seed: int=0):
    rng = np.random.default_rng(seed)
    rows = len(per_class)
    H, W, C = per_class[sorted(per_class.keys())[0]][0].shape
    pad = 2
    grid = np.ones((rows*H + (rows-1)*pad, per_class_n*W + (per_class_n-1)*pad, C), dtype=np.uint8) * 255
    for row, cls in enumerate(sorted(per_class.keys())):
        imgs = per_class[cls]
        idxs = rng.choice(imgs.shape[0], size=min(per_class_n, imgs.shape[0]), replace=False)
        for col, i in enumerate(idxs):
            rs = row*(H+pad)
            cs = col*(W+pad)
            grid[rs:rs+H, cs:cs+W, :] = imgs[i]
    save_image(grid, fig_path)

def save_avg_images_grid(per_class: Dict[int, np.ndarray], fig_path_mean: Path, fig_path_std: Path):
    means = []
    stds = []
    for cls in sorted(per_class.keys()):
        x = per_class[cls].astype(np.float32) / 255.0
        m = (x.mean(axis=0) * 255.0).clip(0,255).astype(np.uint8)
        s = (x.std(axis=0) * 255.0).clip(0,255).astype(np.uint8)
        means.append(m)
        stds.append(s)
    def grid10(imgs: List[np.ndarray]) -> np.ndarray:
        assert len(imgs) == 10
        H, W, C = imgs[0].shape
        pad = 2
        rows, cols = 2, 5
        grid = np.ones((rows*H + (rows-1)*pad, cols*W + (cols-1)*pad, C), dtype=np.uint8) * 255
        idx = 0
        for r in range(rows):
            for c in range(cols):
                rs = r*(H+pad); cs = c*(W+pad)
                grid[rs:rs+H, cs:cs+W, :] = imgs[idx]; idx += 1
        return grid
    save_image(grid10(means), fig_path_mean)
    save_image(grid10(stds), fig_path_std)

def save_pixel_histograms(per_class: Dict[int, np.ndarray], fig_all: Path, fig_per_class: Path=None, bins: int=32):
    X = np.concatenate([imgs.reshape(-1, 3) for imgs in per_class.values()], axis=0)
    plt.figure(figsize=(6,4))
    for ch in range(3):
        plt.hist(X[:, ch], bins=bins, alpha=0.5, density=True, label=f"Channel {ch}")
    plt.legend(frameon=False)
    plt.title("Pixel intensity histogram (global)")
    plt.tight_layout()
    plt.savefig(fig_all, dpi=200)
    plt.close()

    if fig_per_class is not None:
        fig, axes = plt.subplots(2, 5, figsize=(12,5), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, cls in zip(axes, sorted(per_class.keys())):
            x = per_class[cls].reshape(-1, 3)
            for ch in range(3):
                ax.hist(x[:, ch], bins=bins, alpha=0.4, density=True)
            ax.set_title(CIFAR10_LABELS.get(int(cls), str(int(cls))), fontsize=8)
        fig.suptitle("Pixel histograms per class")
        plt.tight_layout()
        plt.savefig(fig_per_class, dpi=200)
        plt.close()

def save_resolution_distribution(per_class: Dict[int, np.ndarray], fig_path: Path):
    sizes = []
    for imgs in per_class.values():
        H, W = imgs.shape[1], imgs.shape[2]
        sizes.append((H, W))
    from collections import Counter
    c = Counter(sizes)
    labels = [f"{h}x{w}" for (h,w) in c.keys()]
    counts = list(c.values())
    plot_bar(counts, labels, "Resolution distribution", fig_path, rotate=0)

def save_augmentations_preview(per_class: Dict[int, np.ndarray], fig_path: Path, seed: int=0):
    if not TORCH_AVAILABLE:
        print("[WARN] torch/torchvision not available; skipping augmentations preview.")
        return
    rng = np.random.default_rng(seed)
    aug_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=1.0),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3),
    ]
    def apply_aug(pil_img, aug):
        t = transforms.Compose([aug])
        return t(pil_img)

    rows = len(per_class); cols = 1 + len(aug_list)
    H, W, _ = per_class[sorted(per_class.keys())[0]][0].shape
    pad = 2
    grid = np.ones((rows*H + (rows-1)*pad, cols*W + (cols-1)*pad, 3), dtype=np.uint8) * 255

    for row, cls in enumerate(sorted(per_class.keys())):
        imgs = per_class[cls]
        i = rng.integers(0, imgs.shape[0])
        base = Image.fromarray(imgs[i])
        grid[row*(H+pad):row*(H+pad)+H, 0:W, :] = np.array(base)
        for j, aug in enumerate(aug_list, start=1):
            aug_img = apply_aug(base, aug)
            grid[row*(H+pad):row*(H+pad)+H, j*(W+pad):j*(W+pad)+W, :] = np.array(aug_img)
    save_image(grid, fig_path)

def save_summary_files(per_class: Dict[int, np.ndarray], fig_dir: Path, extra: Dict = None):
    summary = {}
    total = 0
    class_counts = {}
    per_class_stats = {}
    global_imgs = []
    for cls, imgs in sorted(per_class.items()):
        n = imgs.shape[0]; total += n
        class_counts[cls] = int(n)
        m, s = stats_channel_mean_std(imgs)
        per_class_stats[cls] = {
            "label": CIFAR10_LABELS.get(int(cls), str(int(cls))),
            "mean_rgb": [float(x) for x in m],
            "std_rgb": [float(x) for x in s],
        }
        global_imgs.append(imgs)
    global_imgs = np.concatenate(global_imgs, axis=0)
    gmean, gstd = stats_channel_mean_std(global_imgs)
    summary["total_images"] = int(total)
    summary["class_counts"] = {str(k): v for k, v in class_counts.items()}
    summary["global_mean_rgb"] = [float(x) for x in gmean]
    summary["global_std_rgb"] = [float(x) for x in gstd]
    summary["per_class_stats"] = per_class_stats
    if extra:
        summary.update(extra)
    with open(fig_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# CIFAR-10 Dataset Visualizations",
          "",
          f"- Total images: **{total}**",
          f"- Global mean RGB: **{summary['global_mean_rgb']}**",
          f"- Global std RGB: **{summary['global_std_rgb']}**",
          "",
          "## Figures",
          "",
          "1. **Class counts**: `class_counts.png`",
          "2. **Random samples grid**: `samples_grid.png`",
          "3. **Average images (per class)**: `avg_images_mean.png`, `avg_images_std.png`",
          "4. **Pixel histograms**: `pixel_hist_global.png`, `pixel_hist_per_class.png`",
          "5. **Resolution distribution**: `resolution_distribution.png`",
          "6. **t-SNE (sample features)**: `tsne_samples.png` (if available)",
          "7. **t-SNE (class prototypes)**: `tsne_prototypes.png` (if available)",
          "8. **Augmentations preview**: `augmentations_preview.png` (if available)",
          "" ]
    with open(fig_dir / "README.md", "w") as f:
        f.write("\n".join(md))

def main():
    parser = argparse.ArgumentParser(description="Visualize CIFAR-10 .npy class files and save figures.")
    parser.add_argument("--data_dir", type=str, default="/home/lucaznguyen/FCL/dataset/cifar10-classes", help="Directory containing per-class .npy files.")
    parser.add_argument("--fig_dir", type=str, default="figures/cifar10", help="Output directory for figures.")
    parser.add_argument("--grid_per_class", type=int, default=10, help="#images per class for the random grid (columns).")
    parser.add_argument("--tsne_samples", type=int, default=2000, help="Total images sampled across classes for t-SNE.")
    parser.add_argument("--use_pretrained", action="store_true", help="Use torchvision ResNet18 features for t-SNE/prototypes.")
    parser.add_argument("--augmentations", action="store_true", help="Also save an augmentations preview grid.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    ensure_dir(fig_dir)

    print(f"[INFO] Loading classes from: {data_dir}")
    per_class = load_cifar10_npy_dir(data_dir)
    print(f"[INFO] Loaded {len(per_class)} classes. Example shape: {per_class[sorted(per_class.keys())[0]].shape}")

    class_counts = [per_class[cls].shape[0] for cls in sorted(per_class.keys())]
    labels = [CIFAR10_LABELS.get(int(cls), str(int(cls))) for cls in sorted(per_class.keys())]
    plot_bar(class_counts, labels, "Images per class", fig_dir / "class_counts.png")

    save_random_grid(per_class, fig_dir / "samples_grid.png", per_class_n=args.grid_per_class, seed=args.seed)

    save_avg_images_grid(per_class, fig_dir / "avg_images_mean.png", fig_dir / "avg_images_std.png")

    save_pixel_histograms(per_class, fig_dir / "pixel_hist_global.png", fig_dir / "pixel_hist_per_class.png", bins=32)

    save_resolution_distribution(per_class, fig_dir / "resolution_distribution.png")

    tsne_done = False
    try:
        features, y = compute_features(per_class, args.tsne_samples, args.use_pretrained)
        save_tsne_scatter(features, y, fig_dir / "tsne_samples.png",
                          "t-SNE of sample features" + (" (ResNet18)" if args.use_pretrained else " (PCA→t-SNE on pixels)"))
        tsne_done = True
    except Exception as e:
        print(f"[WARN] Skipping t-SNE on samples: {e}")

    try:
        save_prototypes_tsne(per_class, fig_dir / "tsne_prototypes.png", use_pretrained=args.use_pretrained)
    except Exception as e:
        print(f"[WARN] Skipping prototypes t-SNE: {e}")

    if args.augmentations:
        try:
            save_augmentations_preview(per_class, fig_dir / "augmentations_preview.png", seed=args.seed)
        except Exception as e:
            print(f"[WARN] Skipping augmentations preview: {e}")

    extra = {"tsne_generated": bool(tsne_done), "used_pretrained": bool(args.use_pretrained)}
    save_summary_files(per_class, fig_dir, extra=extra)

    print(f"[DONE] Figures and summary saved to: {fig_dir.resolve()}")

if __name__ == "__main__":
    main()
