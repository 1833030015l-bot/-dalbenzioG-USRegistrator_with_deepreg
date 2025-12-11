from __future__ import annotations

from typing import Dict, Callable, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from monai.data import Dataset
from monai.utils import ensure_tuple

DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {}


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------

def register_dataset(name: str):
    def decorator(fn: Callable[..., Dataset]):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def build_dataset(cfg: dict, split: str, transforms=None) -> Dataset:
    """
    Build a dataset from config.
    """
    name = cfg["name"]

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    factory = DATASET_REGISTRY[name]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}

    return factory(split=split, transforms=transforms, **kwargs)


# -------------------------------------------------------------------------
# Synthetic 3D ellipsoid generator with guaranteed nonzero local variance
# -------------------------------------------------------------------------

class SyntheticEllipsoidsGenerator:
    """
    Produces synthetic ellipsoids with guaranteed nonzero local variance:
    - Main ellipsoid defines shape
    - Soft boundary attenuation preserves shape
    - A spatially smooth random field adds guaranteed textural variation
    """

    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int, int],
        noise_std: float,
        smooth: bool,
        seed: int,
    ):
        self.num_samples = int(num_samples)
        self.image_size = tuple(int(s) for s in image_size)
        self.noise_std = float(noise_std)
        self.smooth = bool(smooth)

        # coordinate grid
        coords = [
            torch.linspace(-1.0, 1.0, steps=s, dtype=torch.float32)
            for s in self.image_size
        ]
        zz, yy, xx = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        self.grid = torch.stack([zz, yy, xx], dim=0)

        # random state
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    # ----------------------------------------------------------
    # Main ellipsoid
    # ----------------------------------------------------------
    def _one_ellipsoid(self) -> torch.Tensor:
        g = self.grid

        cx, cy, cz = self.rng.uniform(-0.25, 0.25, 3)
        rx, ry, rz = self.rng.uniform(0.35, 0.55, 3)

        val = ((g[0] - cx) / rz)**2 + ((g[1] - cy) / ry)**2 + ((g[2] - cz) / rx)**2
        mask = (val <= 1.0).float()

        if self.smooth:
            mask = F.avg_pool3d(mask[None, None], 3, stride=1, padding=1)[0, 0]

        return mask

    # ----------------------------------------------------------
    # Smooth random field guaranteeing local nonzero variance
    # ----------------------------------------------------------
    def _smooth_random_field(self) -> torch.Tensor:
        rnd = torch.rand(self.image_size, dtype=torch.float32)

        # multiple smoothing passes = Gaussian-like effect
        for _ in range(3):
            rnd = F.avg_pool3d(rnd[None, None], 5, stride=1, padding=2)[0, 0]

        # normalize to [0,1]
        rnd -= rnd.min()
        rnd /= rnd.max() + 1e-8

        return rnd

    # ----------------------------------------------------------
    # Final sample
    # ----------------------------------------------------------
    def get_sample(self) -> dict:
        base = self._one_ellipsoid()
        texture = self._smooth_random_field()

        # amplify texture near boundaries to guarantee LNCC signal
        boundary = F.conv3d(
            base[None, None],
            weight=torch.ones(1, 1, 5, 5, 5),
            padding=2
        )[0, 0]
        boundary = (boundary > 0) & (boundary < 125)  # inside 5×5×5 window
        boundary = boundary.float()

        # mix components:
        # - inside ellipsoid: mild randomness
        # - boundary: stronger randomness
        # - outside: weak randomness
        img = (
            base * (0.7 + 0.3 * texture) +
            boundary * 0.4 * texture +
            (1 - base) * 0.1 * texture
        )

        # additional global noise
        img = img + torch.randn_like(img) * self.noise_std
        img = img.clamp(0.0, 1.0)

        return {
            "moving": img[None],
            "fixed": img[None].clone(),  # symmetrically defined; model must warp
        }


# -------------------------------------------------------------------------
# MONAI Dataset wrapper
# -------------------------------------------------------------------------

class SyntheticEllipsoidsMonaiDataset(Dataset):
    def __init__(self, generator: SyntheticEllipsoidsGenerator, transforms=None):
        self.generator = generator
        self.transforms = transforms
        self.data = [{"idx": i} for i in range(len(generator))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.generator.get_sample()
        if self.transforms:
            sample = self.transforms(sample)
        return sample


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

@register_dataset("synthetic_ellipsoids")
def create_synthetic_ellipsoids(
    split: str,
    num_samples: int,
    image_size=(64, 64, 64),
    noise_std: float = 0.03,
    smooth: bool = True,
    seed: int = 123,
    transforms=None,
):
    if split.lower() == "train":
        s = seed
    elif split.lower() == "val":
        s = seed + 1
    else:
        s = seed + 2

    generator = SyntheticEllipsoidsGenerator(
        num_samples=num_samples,
        image_size=tuple(image_size),
        noise_std=noise_std,
        smooth=smooth,
        seed=s,
    )

    return SyntheticEllipsoidsMonaiDataset(generator, transforms=transforms)


# -------------------------------------------------------------------------
# DeepReg-style synthetic DVF dataset
# -------------------------------------------------------------------------


class DeepRegLikeDVFSyntheticGenerator:
    """
    DeepReg 风格的 synthetic DVF 生成器（PyTorch 版本）

    逻辑：
    - 用 SyntheticEllipsoidsGenerator 生成一个 base 体数据作为 fixed
    - 在低分辨率网格上生成 3 通道高斯随机场（low_res_field）
    - 用 U(0, max_disp) 的随机强度缩放 → 控制整体形变幅度
    - 通过三线性插值上采样到 full resolution → 光滑的随机 DVF
    - identity_grid + DVF → grid_sample 得到 moving
    - 返回 (moving, fixed, dvf)
    """

    def __init__(
        self,
        num_samples: int,
        image_size: Tuple[int, int, int],
        max_disp: float = 0.2,
        cp_spacing: int = 8,
        noise_std: float = 0.03,
        smooth: bool = True,
        seed: int = 123,
    ):
        self.num_samples = int(num_samples)
        self.image_size = tuple(int(s) for s in image_size)
        self.max_disp = float(max_disp)
        self.cp_spacing = int(cp_spacing)

        # 复用原来的椭球体生成器：相当于 DeepReg 的 anatomy
        self.base_generator = SyntheticEllipsoidsGenerator(
            num_samples=num_samples,
            image_size=self.image_size,
            noise_std=noise_std,
            smooth=smooth,
            seed=seed,
        )

        self.rng = np.random.RandomState(seed)

        # 预先创建归一化坐标的 identity grid (1, D, H, W, 3)
        D, H, W = self.image_size
        zz = torch.linspace(-1.0, 1.0, steps=D, dtype=torch.float32)
        yy = torch.linspace(-1.0, 1.0, steps=H, dtype=torch.float32)
        xx = torch.linspace(-1.0, 1.0, steps=W, dtype=torch.float32)
        z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
        self.identity_grid = torch.stack([z, y, x], dim=-1)[None]

    def _random_dvf(self) -> torch.Tensor:
        """
        基于 DeepReg 的 gen_rand_ddf 思路生成随机 DDF:

        - 在 low_res_size 上生成 3D 高斯随机场
        - 用 U(0, max_disp) 的随机强度缩放
        - F.interpolate 三线性插值到 full resolution
        - 返回 (1, D, H, W, 3) 的 DVF（与 grid_sample 归一化坐标匹配）
        """
        D, H, W = self.image_size

        # 低分辨率网格尺寸（对应 low_res_size）
        Dc = max(1, D // self.cp_spacing)
        Hc = max(1, H // self.cp_spacing)
        Wc = max(1, W // self.cp_spacing)

        # 随机场强度 U(0, max_disp)，每通道一个
        low_res_strength = self.rng.uniform(
            low=0.0,
            high=self.max_disp,
            size=(1, 1, 1, 1, 3),
        ).astype(np.float32)

        # 低分辨率高斯场 (1, Dc, Hc, Wc, 3)
        low_res_field = self.rng.randn(1, Dc, Hc, Wc, 3).astype(np.float32)
        low_res_field = low_res_field * low_res_strength  # 广播

        # 转 tensor: (1, 3, Dc, Hc, Wc)
        low_res_field = torch.from_numpy(low_res_field)
        low_res_field = low_res_field.permute(0, 4, 1, 2, 3)

        # 上采样到 full res: (1, 3, D, H, W)
        dvf_full = F.interpolate(
            low_res_field,
            size=(D, H, W),
            mode="trilinear",
            align_corners=True,
        )

        # 转成 grid_sample 的 (1, D, H, W, 3)
        dvf_grid = dvf_full.permute(0, 2, 3, 4, 1)
        return dvf_grid

    def __len__(self) -> int:
        return self.num_samples

    def get_sample(self) -> dict:
        # 取一个椭球体样本作为 fixed（(1,D,H,W)）
        base = self.base_generator.get_sample()
        fixed = base["fixed"].unsqueeze(0)  # (1,1,D,H,W)

        dvf_grid = self._random_dvf()       # (1,D,H,W,3)
        grid = self.identity_grid + dvf_grid

        # 用 DVF warp 出 moving
        moving = F.grid_sample(
            fixed,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1,1,D,H,W)

        # 通道优先 DVF (3,D,H,W)
        dvf_ch_first = dvf_grid[0].permute(3, 0, 1, 2)

        return {
            "moving": moving[0],   # (1,D,H,W)
            "fixed": fixed[0],     # (1,D,H,W)
            "dvf": dvf_ch_first,   # (3,D,H,W)
        }


@register_dataset("deepreg_synthetic")
def create_deepreg_synthetic(
    split: str,
    num_samples: int,
    image_size=(64, 64, 64),
    max_disp: float = 0.2,
    cp_spacing: int = 8,
    noise_std: float = 0.03,
    smooth: bool = True,
    seed: int = 123,
    transforms=None,
):
    """
    DeepReg-style synthetic DVF dataset 工厂。

    YAML 示例:

        train_dataset:
          name: deepreg_synthetic
          image_size: [64, 64, 64]
          num_samples: 4000
          max_disp: 0.2
          cp_spacing: 8
          noise_std: 0.03
          smooth: true
          seed: 123
    """
    # 不同 split 改个 seed
    if split.lower() == "train":
        s = seed
    elif split.lower() == "val":
        s = seed + 1
    else:
        s = seed + 2

    generator = DeepRegLikeDVFSyntheticGenerator(
        num_samples=num_samples,
        image_size=tuple(image_size),
        max_disp=max_disp,
        cp_spacing=cp_spacing,
        noise_std=noise_std,
        smooth=smooth,
        seed=s,
    )

    # 复用原来的 MONAI Dataset 封装
    return SyntheticEllipsoidsMonaiDataset(generator, transforms=transforms)

