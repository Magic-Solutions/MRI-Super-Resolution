"""Display a 3D MRI volume as an animated slice-by-slice video.

Loads a NIfTI file and animates through slices along a chosen axis using
matplotlib.  Supports axial, coronal, and sagittal views, or all three
side-by-side.

Usage:
    python view_mri.py <nifti_path> [--axis {axial,coronal,sagittal,all}]
                                     [--fps 15] [--save out.mp4]

Examples:
    python view_mri.py ../FOMO60K/PT001_OASIS1/sub_1/ses_1/t1.nii.gz
    python view_mri.py ../FOMO60K/PT001_OASIS1/sub_1/ses_1/t1.nii.gz --axis all
    python view_mri.py ../FOMO60K/PT001_OASIS1/sub_1/ses_1/t1.nii.gz --save scan.mp4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import FuncAnimation

AXIS_MAP = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = img.header.get_zooms()
    return data, zooms


def get_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    slc = np.take(volume, idx, axis=axis)
    return np.rot90(slc)


def animate_single_axis(
    volume: np.ndarray,
    axis_name: str,
    fps: int,
    save_path: Path | None,
    title: str,
) -> None:
    axis = AXIS_MAP[axis_name]
    n_slices = volume.shape[axis]
    sample = get_slice(volume, axis, 0)
    h, w = sample.shape
    scale = 6.0 / max(h, w)
    fig_w, fig_h = w * scale + 0.5, h * scale + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")

    im = ax.imshow(sample, cmap="gray", aspect="equal")
    label = ax.set_title(
        f"{title}\n{axis_name} slice 0/{n_slices - 1}",
        color="white",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    def update(frame: int) -> list:
        idx = frame % n_slices
        im.set_data(get_slice(volume, axis, idx))
        label.set_text(f"{title}\n{axis_name} slice {idx}/{n_slices - 1}")
        return [im, label]

    anim = FuncAnimation(
        fig, update, frames=n_slices, interval=1000 // fps, blit=True
    )

    if save_path:
        anim.save(str(save_path), writer="ffmpeg", fps=fps, dpi=120)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def animate_all_axes(
    volume: np.ndarray,
    fps: int,
    save_path: Path | None,
    title: str,
) -> None:
    names = ["sagittal", "coronal", "axial"]
    axes = [AXIS_MAP[n] for n in names]
    n_slices = [volume.shape[a] for a in axes]
    max_slices = max(n_slices)

    samples = [get_slice(volume, a, volume.shape[a] // 2) for a in axes]
    widths = [s.shape[1] for s in samples]
    max_h = max(s.shape[0] for s in samples)
    total_w = sum(widths)
    scale = 16.0 / total_w
    fig_h = max_h * scale + 1.2

    fig, axs = plt.subplots(
        1, 3,
        figsize=(16, fig_h),
        gridspec_kw={"width_ratios": widths},
    )
    fig.patch.set_facecolor("black")
    fig.suptitle(title, color="white", fontsize=13, y=0.98)

    ims = []
    labels = []
    for i, (name, ax_idx) in enumerate(zip(names, axes)):
        axs[i].set_facecolor("black")
        axs[i].axis("off")
        im = axs[i].imshow(samples[i], cmap="gray", aspect="equal")
        lbl = axs[i].set_title(
            f"{name}  0/{n_slices[i] - 1}", color="white", fontsize=11
        )
        ims.append(im)
        labels.append(lbl)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.05)

    def update(frame: int) -> list:
        artists = []
        for i, (name, ax_idx, ns) in enumerate(zip(names, axes, n_slices)):
            idx = frame % ns
            ims[i].set_data(get_slice(volume, ax_idx, idx))
            labels[i].set_text(f"{name}  {idx}/{ns - 1}")
            artists += [ims[i], labels[i]]
        return artists

    anim = FuncAnimation(
        fig, update, frames=max_slices, interval=1000 // fps, blit=True
    )

    if save_path:
        anim.save(str(save_path), writer="ffmpeg", fps=fps, dpi=120)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a 3D MRI NIfTI volume as a slice animation."
    )
    parser.add_argument("nifti", type=Path, help="Path to .nii or .nii.gz file")
    parser.add_argument(
        "--axis",
        choices=["axial", "coronal", "sagittal", "all"],
        default="axial",
        help="Slice orientation (default: axial)",
    )
    parser.add_argument(
        "--fps", type=int, default=15, help="Frames per second (default: 15)"
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save animation to file instead of displaying (e.g. out.mp4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.nifti.exists():
        raise FileNotFoundError(f"File not found: {args.nifti}")

    print(f"Loading {args.nifti} ...")
    volume, zooms = load_volume(args.nifti)
    print(f"  Shape: {volume.shape}  Voxel size: {zooms}")

    vmin, vmax = np.percentile(volume[volume > 0], [1, 99])
    volume = np.clip(volume, vmin, vmax)

    title = args.nifti.stem.replace(".nii", "")

    if args.axis == "all":
        animate_all_axes(volume, args.fps, args.save, title)
    else:
        animate_single_axis(volume, args.axis, args.fps, args.save, title)


if __name__ == "__main__":
    main()
