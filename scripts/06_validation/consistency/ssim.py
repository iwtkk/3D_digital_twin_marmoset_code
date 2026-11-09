import argparse
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def ssim_components_channel(img1, img2, L=255, K1=0.01, K2=0.03, win_sigma=1.5):
    """
    Compute per-component SSIM maps for a pair of single-channel images
    Returns:
        l_map, c_map, s_map, ssim_map
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, win_sigma)
    mu2 = gaussian_filter(img2, win_sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, win_sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, win_sigma) - mu2_sq
    sigma12  = gaussian_filter(img1 * img2, win_sigma) - mu1_mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2.0

    l_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0))
    c_map = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)

    denom_s = (sigma1 * sigma2 + C3)
    s_map = (sigma12 + C3) / denom_s
    s_map[denom_s == 0] = 1

    ssim_map = l_map * c_map * s_map

    return l_map, c_map, s_map, ssim_map


def ssim_rgb_full_maps(img1_bgr, img2_bgr, win_sigma=1.5):
    if img1_bgr.shape != img2_bgr.shape:
        raise ValueError("Image sizes differ")

    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    R1, G1, B1 = cv2.split(img1)
    R2, G2, B2 = cv2.split(img2)

    result = {}

    for name, ch1, ch2 in [
        ("R", R1, R2),
        ("G", G1, G2),
        ("B", B1, B2),
    ]:
        l_map, c_map, s_map, ssim_map = ssim_components_channel(
            ch1, ch2, L=255, win_sigma=win_sigma
        )

        result[name] = {
            "l": l_map,
            "c": c_map,
            "s": s_map,
            "ssim_map": ssim_map,
            "ssim": float(ssim_map.mean()),
        }

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SSIM and component maps (luminance/contrast/structure) for two RGB images."
    )
    # Two required image paths
    parser.add_argument("img1", help="Path to the first image")
    parser.add_argument("img2", help="Path to the second image")

    # Optional arguments
    parser.add_argument(
        "--win-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for SSIM (default: 1.5)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show R-channel maps with matplotlib",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load images
    img1 = cv2.imread(args.img1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(args.img2, cv2.IMREAD_COLOR)

    if img1 is None:
        raise FileNotFoundError(f"Could not read img1: {args.img1}")
    if img2 is None:
        raise FileNotFoundError(f"Could not read img2: {args.img2}")

    result = ssim_rgb_full_maps(img1, img2, win_sigma=args.win_sigma)

    print("== SSIM scores ==")
    print(f"R channel SSIM: {result['R']['ssim']:.4f}")
    print(f"G channel SSIM: {result['G']['ssim']:.4f}")
    print(f"B channel SSIM: {result['B']['ssim']:.4f}")

    if args.show:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(result["R"]["l"], cmap="jet")
        axes[0].set_title("R - Luminance")

        axes[1].imshow(result["R"]["c"], cmap="jet")
        axes[1].set_title("R - Contrast")

        axes[2].imshow(result["R"]["s"], cmap="jet")
        axes[2].set_title("R - Structure")

        im = axes[3].imshow(result["R"]["ssim_map"], cmap="jet", vmin=0, vmax=1)
        axes[3].set_title("R - SSIM Map")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
