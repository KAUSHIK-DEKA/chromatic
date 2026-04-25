"""
color_extractor.py

Core color extraction and classification logic.
Two extraction modes:
  - extract_garment_color: for clothing (skirts, co-ords, tops, dresses)
  - extract_footwear_color: for shoes (handles skin, busy backgrounds)

classify_color: maps an RGB tuple to (specific_color, color_family).
"""
import colorsys
import numpy as np
from PIL import Image


def rgb_to_hsv_deg(rgb):
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


def classify_color(rgb):
    """Return (specific_color, color_family) for the given RGB tuple."""
    r, g, b = rgb
    h, s, v = rgb_to_hsv_deg(rgb)

    if v < 0.17:
        if s > 0.25 and r > b + 15:
            return ("Chocolate", "Brown")
        return ("Black", "Black")

    if v < 0.30:
        if s < 0.15:
            return ("Charcoal", "Gray")
        if 10 <= h < 30:
            return ("Chocolate", "Brown")
        if h < 10 or h >= 345:
            return ("Burgundy", "Red")
        if 200 <= h < 255:
            return ("Navy", "Blue")
        if 255 <= h < 300:
            return ("Eggplant", "Purple")
        if 85 <= h < 170:
            return ("Forest Green", "Green")
        if 30 <= h < 65:
            return ("Olive Green", "Green")

    if v >= 0.85 and s < 0.14:
        if r > b + 10:
            if v > 0.92 and s < 0.06:
                return ("Ivory", "White")
            if s > 0.06:
                return ("Cream", "White")
            return ("Off-White", "White")
        return ("White", "White")

    if v > 0.70 and s < 0.14:
        if r > b + 10:
            if v > 0.85:
                return ("Cream", "White")
            if r > b + 15:
                return ("Beige", "Beige")
            return ("Off-White", "White")
        if g > r and g > b and (g - r) >= 3:
            return ("Sage", "Green")
        return ("Light Gray", "Gray")

    if s < 0.10:
        if v > 0.55:
            return ("Gray", "Gray")
        elif v > 0.28:
            return ("Dark Gray", "Gray")
        else:
            return ("Charcoal", "Gray")

    if s < 0.20 and v > 0.55:
        if g >= r and g >= b and (g - min(r, b)) >= 8:
            if v > 0.70:
                return ("Sage", "Green")
            return ("Olive", "Green")
        if r > g > b and (r - b) >= 10:
            if v > 0.80:
                return ("Beige", "Beige")
            if v > 0.60:
                return ("Nude", "Beige")
            return ("Taupe", "Brown")
        if b > r and b > g:
            if v > 0.70:
                return ("Powder Blue", "Blue")
            return ("Denim Blue", "Blue")

    def in_hue(low, high):
        if low < high:
            return low <= h < high
        return h >= low or h < high

    if in_hue(345, 360) or in_hue(0, 10):
        if v < 0.32:
            return ("Burgundy", "Red")
        if v < 0.45 and s > 0.50:
            return ("Wine", "Red")
        if v < 0.60 and s > 0.55:
            return ("Crimson", "Red")
        if v > 0.75 and s < 0.30:
            if v > 0.85:
                return ("Baby Pink", "Pink")
            return ("Blush", "Pink")
        if v > 0.60 and s < 0.40:
            return ("Rose", "Pink")
        if s > 0.55:
            return ("Red", "Red")
        return ("Rose", "Pink")

    if 10 <= h < 40:
        if v < 0.38:
            return ("Chocolate", "Brown")
        if v < 0.50:
            if s > 0.70 and h < 18 and v > 0.42:
                return ("Rust", "Red")
            if s > 0.40:
                return ("Brown", "Brown")
            return ("Coffee", "Brown")
        if v < 0.62:
            if s > 0.62 and h < 20:
                if h < 17 and s > 0.70:
                    return ("Rust", "Red")
                return ("Terracotta", "Orange")
            if s > 0.45:
                return ("Camel", "Brown")
            if s > 0.25:
                return ("Mocha", "Brown")
            return ("Taupe", "Brown")
        if v < 0.75:
            if s > 0.60:
                return ("Burnt Orange", "Orange")
            if s > 0.40:
                return ("Tan", "Brown")
            if s > 0.22:
                return ("Nude", "Beige")
            return ("Beige", "Beige")
        if s > 0.55:
            return ("Orange", "Orange")
        if s > 0.30:
            return ("Peach", "Pink")
        if s > 0.18:
            return ("Nude", "Beige")
        return ("Beige", "Beige")

    if 40 <= h < 65:
        if v < 0.42:
            return ("Olive", "Green")
        if v < 0.58 and s > 0.45:
            return ("Mustard", "Yellow")
        if s < 0.22:
            if v > 0.85:
                return ("Cream", "White")
            if v > 0.70:
                return ("Beige", "Beige")
            return ("Sage", "Green")
        if v > 0.85:
            if s < 0.50:
                return ("Butter Yellow", "Yellow")
            return ("Pastel Yellow", "Yellow")
        if s > 0.55:
            return ("Yellow", "Yellow")
        return ("Mustard", "Yellow")

    if 65 <= h < 95:
        if v < 0.45:
            return ("Olive Green", "Green")
        if s < 0.25:
            return ("Sage", "Green")
        if s > 0.55 and v > 0.70:
            return ("Lime", "Green")
        return ("Olive", "Green")

    if 95 <= h < 165:
        if v < 0.32:
            return ("Forest Green", "Green")
        if v < 0.45 and s > 0.40:
            return ("Dark Green", "Green")
        if v > 0.85 and s < 0.30:
            return ("Mint", "Green")
        if s < 0.25:
            return ("Sage", "Green")
        if s > 0.55 and v < 0.55:
            return ("Emerald", "Green")
        return ("Green", "Green")

    if 165 <= h < 200:
        if s > 0.45:
            return ("Teal", "Green")
        if v > 0.85:
            return ("Powder Blue", "Blue")
        return ("Turquoise", "Blue")

    if 200 <= h < 255:
        if v < 0.28:
            return ("Navy", "Blue")
        if v < 0.42:
            if s > 0.55:
                return ("Royal Blue", "Blue")
            return ("Navy", "Blue")
        if s < 0.25:
            if v > 0.85:
                return ("Powder Blue", "Blue")
            return ("Denim Blue", "Blue")
        if v > 0.85 and s < 0.45:
            return ("Baby Blue", "Blue")
        if v > 0.72 and s < 0.55:
            return ("Sky Blue", "Blue")
        if s > 0.70 and v < 0.55:
            return ("Cobalt", "Blue")
        if s > 0.50:
            return ("Royal Blue", "Blue")
        return ("Denim Blue", "Blue")

    if 255 <= h < 295:
        if v < 0.32:
            return ("Eggplant", "Purple")
        if v < 0.48:
            return ("Plum", "Purple")
        if v > 0.80 and s < 0.35:
            return ("Lavender", "Purple")
        if v > 0.75 and s < 0.50:
            return ("Lilac", "Purple")
        if s > 0.50:
            return ("Purple", "Purple")
        return ("Violet", "Purple")

    if 295 <= h < 345:
        if v < 0.35:
            return ("Plum", "Purple")
        if v < 0.50 and s > 0.55:
            return ("Magenta", "Pink")
        if v > 0.85 and s < 0.30:
            return ("Baby Pink", "Pink")
        if v > 0.75 and s < 0.35:
            return ("Blush", "Pink")
        if s > 0.55:
            if v > 0.60:
                return ("Hot Pink", "Pink")
            return ("Fuchsia", "Pink")
        if s < 0.40:
            return ("Mauve", "Pink")
        if v > 0.70:
            return ("Pink", "Pink")
        return ("Rose", "Pink")

    return ("Unknown", "Unknown")


def extract_garment_color(img_path_or_bytes):
    """Extract dominant color from a garment image (skirt, dress, top, co-ord)."""
    img = Image.open(img_path_or_bytes).convert('RGB')
    arr = np.array(img)
    h, w, _ = arr.shape

    regions = [
        arr[int(h*0.30):int(h*0.45), int(w*0.38):int(w*0.62)],
        arr[int(h*0.45):int(h*0.60), int(w*0.35):int(w*0.65)],
        arr[int(h*0.55):int(h*0.68), int(w*0.38):int(w*0.62)],
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
        neutral = ((abs(r_ch - g_ch) < 8) & (abs(g_ch - b_ch) < 8)
                   & (abs(r_ch - b_ch) < 10) & (r_ch > 150) & (r_ch < 230))
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
    img = Image.open(img_path_or_bytes).convert('RGB')
    arr = np.array(img)
    h, w, _ = arr.shape

    regions = [
        arr[int(h*0.35):int(h*0.55), int(w*0.20):int(w*0.50)],
        arr[int(h*0.35):int(h*0.55), int(w*0.50):int(w*0.80)],
        arr[int(h*0.55):int(h*0.80), int(w*0.20):int(w*0.50)],
        arr[int(h*0.55):int(h*0.80), int(w*0.50):int(w*0.80)],
        arr[int(h*0.45):int(h*0.75), int(w*0.35):int(w*0.65)],
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
        neutral = ((abs(r_ch - g_ch) < 8) & (abs(g_ch - b_ch) < 8)
                   & (abs(r_ch - b_ch) < 10) & (r_ch > 150) & (r_ch < 230))
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


def process_image(img_source, mode='garment'):
    """Extract and classify color from one image.
    Returns dict: {family, specific, rgb, hex}
    """
    extractor = extract_footwear_color if mode == 'footwear' else extract_garment_color
    rgb = extractor(img_source)
    specific, family = classify_color(rgb)
    return {
        'family': family,
        'specific': specific,
        'rgb': rgb,
        'hex': f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
    }
