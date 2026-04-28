"""
color_extractor.py

Color extraction + classification.

Pipeline:
  1. Sample dominant garment/footwear color from image (RGB)
  2. Convert to CIELAB
  3. Find nearest canonical color in palette via Delta E (CIE76)
  4. Return canonical name, canonical hex code, and parent color

The palette is loaded from palette.json (~285 named colors with
established hex codes and parent color groupings, derived from the
brand's existing taxonomy).
"""
import colorsys
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------- Palette loading ----------------
_PALETTE_PATH = Path(__file__).parent / "palette.json"

with open(_PALETTE_PATH, "r") as _f:
    PALETTE = json.load(_f)


def _hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# Pre-compute palette RGB and Lab arrays for fast lookup
_PALETTE_RGB = np.array([_hex_to_rgb(e["hex"]) for e in PALETTE], dtype=float)
_PALETTE_NAMES = [e["name"] for e in PALETTE]
_PALETTE_HEXES = [e["hex"] for e in PALETTE]
_PALETTE_PARENTS = [e["parent"] for e in PALETTE]


# ---------------- sRGB -> CIELAB ----------------
def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb):
    """Convert sRGB (0-255) to CIELAB. Accepts shape (3,) or (N, 3)."""
    arr = np.asarray(rgb, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]

    lin = _srgb_to_linear(arr)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ M.T
    xn, yn, zn = 0.95047, 1.0, 1.08883
    xyz_n = xyz / np.array([xn, yn, zn])
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz_n > eps, np.cbrt(xyz_n), (kappa * xyz_n + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b_ = 200 * (f[:, 1] - f[:, 2])
    out = np.stack([L, a, b_], axis=-1)
    return out[0] if single else out


_PALETTE_LAB = rgb_to_lab(_PALETTE_RGB)


def rgb_to_hsv_deg(rgb):
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


# ---------------- Classification ----------------
def classify_color(rgb):
    """Return dict with keys: name, hex, parent.

    Uses Delta E (CIE76) in CIELAB to find the perceptually nearest
    canonical color from the palette.
    """
    rgb_arr = np.array(rgb, dtype=float)
    lab = rgb_to_lab(rgb_arr)
    deltas = np.sqrt(np.sum((_PALETTE_LAB - lab) ** 2, axis=1))
    idx = int(np.argmin(deltas))

    return {
        "name": _PALETTE_NAMES[idx],
        "hex": _PALETTE_HEXES[idx],
        "parent": _PALETTE_PARENTS[idx],
    }


# ---------------- Image color extractors ----------------
def extract_garment_color(img_path_or_bytes):
    """Extract dominant color from a garment image (skirt/dress/top/co-ord)."""
    img = Image.open(img_path_or_bytes).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    regions = [
        arr[int(h * 0.30): int(h * 0.45), int(w * 0.38): int(w * 0.62)],
        arr[int(h * 0.45): int(h * 0.60), int(w * 0.35): int(w * 0.65)],
        arr[int(h * 0.55): int(h * 0.68), int(w * 0.38): int(w * 0.62)],
    ]
    all_px = np.vstack([r.reshape(-1, 3) for r in regions])
    r_ch = all_px[:, 0].astype(int)
    g_ch = all_px[:, 1].astype(int)
    b_ch = all_px[:, 2].astype(int)

    pre_median = np.median(all_px, axis=0)
    pm_min = pre_median.min()
    pm_max = pre_median.max()
    product_is_white = all(c > 215 for c in pre_median)
    product_is_gray = (pm_max - pm_min) < 8 and 180 < pm_min < 225

    kill = []
    if not product_is_white:
        kill.append((r_ch > 228) & (g_ch > 228) & (b_ch > 228))
    if not product_is_gray and not product_is_white:
        neutral = (
            (abs(r_ch - g_ch) < 8) & (abs(g_ch - b_ch) < 8)
            & (abs(r_ch - b_ch) < 10) & (r_ch > 150) & (r_ch < 230)
        )
        kill.append(neutral)

    if kill:
        m = np.zeros(len(all_px), dtype=bool)
        for k in kill:
            m |= k
        keep = ~m
        if keep.sum() > 500:
            all_px = all_px[keep]

    final = np.median(all_px, axis=0).astype(int)
    return tuple(int(c) for c in final)


def extract_footwear_color(img_path_or_bytes):
    """Extract dominant color from a footwear image.
    Handles feet/legs, busy lifestyle backgrounds, flat-lays.
    """
    img = Image.open(img_path_or_bytes).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    regions = [
        arr[int(h * 0.35): int(h * 0.55), int(w * 0.20): int(w * 0.50)],
        arr[int(h * 0.35): int(h * 0.55), int(w * 0.50): int(w * 0.80)],
        arr[int(h * 0.55): int(h * 0.80), int(w * 0.20): int(w * 0.50)],
        arr[int(h * 0.55): int(h * 0.80), int(w * 0.50): int(w * 0.80)],
        arr[int(h * 0.45): int(h * 0.75), int(w * 0.35): int(w * 0.65)],
    ]
    all_px = np.vstack([r.reshape(-1, 3) for r in regions])
    r_ch = all_px[:, 0].astype(int)
    g_ch = all_px[:, 1].astype(int)
    b_ch = all_px[:, 2].astype(int)

    pre_median = np.median(all_px, axis=0)
    pm_min = pre_median.min()
    pm_max = pre_median.max()
    product_is_white = all(c > 215 for c in pre_median)
    product_is_gray = (pm_max - pm_min) < 8 and 180 < pm_min < 225

    kill_masks = []
    if not product_is_white:
        kill_masks.append((r_ch > 225) & (g_ch > 225) & (b_ch > 225))
    if not product_is_gray and not product_is_white:
        neutral = (
            (abs(r_ch - g_ch) < 8) & (abs(g_ch - b_ch) < 8)
            & (abs(r_ch - b_ch) < 10) & (r_ch > 150) & (r_ch < 230)
        )
        kill_masks.append(neutral)

    skin = (
        (r_ch > g_ch) & (g_ch > b_ch)
        & ((r_ch - b_ch) > 25) & ((r_ch - b_ch) < 100)
        & (r_ch > 130) & (r_ch < 245)
        & ((r_ch - g_ch) <= (r_ch - b_ch))
        & (abs((g_ch - b_ch) - (r_ch - g_ch)) < 30)
    )
    if skin.sum() < len(all_px) * 0.65:
        kill_masks.append(skin)

    if kill_masks:
        m = np.zeros(len(all_px), dtype=bool)
        for k in kill_masks:
            m |= k
        keep = ~m
        if keep.sum() > 400:
            all_px = all_px[keep]

    if len(all_px) == 0:
        return (128, 128, 128)

    brightness = all_px.max(axis=1)
    brightness_std = brightness.std()
    px_arr = all_px.astype(float) / 255.0
    px_max = px_arr.max(axis=1)
    px_min = px_arr.min(axis=1)
    sat = np.where(px_max > 0, (px_max - px_min) / np.maximum(px_max, 1e-9), 0)
    sat_std = sat.std()
    is_busy = (brightness_std > 55) and (sat_std > 0.18)

    median_rgb = np.median(all_px, axis=0).astype(int)

    dark_mask = brightness < 60
    dark_frac = dark_mask.sum() / len(all_px)

    if is_busy and dark_frac > 0.15:
        dark_px = all_px[dark_mask]
        final = np.median(dark_px, axis=0).astype(int)
    else:
        final = median_rgb

    return tuple(int(c) for c in final)


def process_image(img_source, mode="garment"):
    """Extract and classify color from one image.
    Returns dict: {name, hex, parent, sampled_rgb, sampled_hex}
    """
    extractor = extract_footwear_color if mode == "footwear" else extract_garment_color
    rgb = extractor(img_source)
    classified = classify_color(rgb)
    return {
        "name": classified["name"],
        "hex": classified["hex"],
        "parent": classified["parent"],
        "sampled_rgb": rgb,
        "sampled_hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
    }
