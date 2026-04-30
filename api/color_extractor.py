"""
color_extractor.py

Color extraction + classification with human-in-the-loop calibration.

Pipeline:
  1. Sample dominant garment/footwear color from image (RGB).
  2. Hash the image — if we have a stored correction for this exact image,
     return it directly (image-hash override).
  3. Otherwise convert sampled RGB to CIELAB, find perceptually nearest
     entry in the augmented palette via Delta E (CIE76).
  4. Augmented palette = original palette.json + user_anchors from
     corrections.json. Each user correction becomes a new palette anchor.

Corrections store schema (corrections.json):
  {
    "image_overrides": {
      "<sha256_of_image_bytes>": {
        "name": "Midnight Blue",
        "hex": "#0F2740",
        "parent": "Blue",
        "sampled_rgb": [r, g, b],
        "ts": "2026-04-29T...",
      }
    },
    "user_anchors": [
      {
        "name": "Midnight Blue",
        "hex": "#0F2740",
        "parent": "Blue",
        "rgb": [r, g, b],
        "weight": 1,
        "ts": "...",
      },
      ...
    ]
  }

The augmented palette = palette.json entries + user_anchors entries.
When a user corrects an image, we add the *sampled RGB* (the color we
read from the image) -> the corrected name as an anchor. Next time any
image has a dominant RGB near that point, the user's chosen name will
be the nearest neighbour and will win.
"""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------------- Paths ----------------
_API_DIR = Path(__file__).parent
_PALETTE_PATH = _API_DIR / "palette.json"

# Corrections file lives in a writable directory. On Render's free tier the
# whole filesystem is writable but ephemeral (wiped on redeploy). Users can
# export via /corrections/export and re-import via /corrections/import.
_CORRECTIONS_DIR_ENV = os.environ.get("CHROMATIC_DATA_DIR")
if _CORRECTIONS_DIR_ENV:
    _DATA_DIR = Path(_CORRECTIONS_DIR_ENV)
else:
    _DATA_DIR = _API_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CORRECTIONS_PATH = _DATA_DIR / "corrections.json"


# ---------------- Palette + corrections loading ----------------
with open(_PALETTE_PATH, "r", encoding="utf-8") as _f:
    BASE_PALETTE = json.load(_f)


def _hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _empty_corrections():
    return {"image_overrides": {}, "user_anchors": []}


def _load_corrections():
    if not _CORRECTIONS_PATH.exists():
        return _empty_corrections()
    try:
        with open(_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Sanity: ensure both keys exist
        data.setdefault("image_overrides", {})
        data.setdefault("user_anchors", [])
        return data
    except Exception:
        return _empty_corrections()


# Mutex protects in-memory state and the corrections file
_lock = threading.Lock()
_corrections = _load_corrections()


# ---------------- Augmented palette (rebuilt on every correction) ----------------
_palette_names: list = []
_palette_hexes: list = []
_palette_parents: list = []
_palette_lab: np.ndarray = np.zeros((0, 3))


def _rebuild_palette_index():
    """Re-compute the cached arrays for fast nearest-neighbour lookup.

    Combines BASE_PALETTE entries with user_anchors. Anchors with higher
    weight get inserted multiple times so they pull harder (Delta E is
    distance-based, but if multiple entries collide at the same point,
    argmin still picks the first — duplication doesn't change winners,
    so weight is informational for now and reserved for future
    weighted-distance variants).
    """
    global _palette_names, _palette_hexes, _palette_parents, _palette_lab

    entries = []
    for e in BASE_PALETTE:
        rgb = _hex_to_rgb(e["hex"]) if "hex" in e and not isinstance(e.get("rgb"), list) else tuple(e["rgb"])
        entries.append({
            "name": e["name"],
            "hex": e["hex"],
            "parent": e["parent"],
            "rgb": rgb,
        })

    for anchor in _corrections.get("user_anchors", []):
        entries.append({
            "name": anchor["name"],
            "hex": anchor["hex"],
            "parent": anchor["parent"],
            "rgb": tuple(anchor["rgb"]),
        })

    _palette_names = [e["name"] for e in entries]
    _palette_hexes = [e["hex"] for e in entries]
    _palette_parents = [e["parent"] for e in entries]
    rgbs = np.array([e["rgb"] for e in entries], dtype=float)
    _palette_lab = rgb_to_lab(rgbs)


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


# Now that rgb_to_lab exists, build the index for the first time
_rebuild_palette_index()


# ---------------- Classification ----------------
def classify_color(rgb):
    """Map an RGB tuple to a palette entry. Returns dict {name, hex, parent}."""
    lab = rgb_to_lab(np.asarray(rgb, dtype=float))
    deltas = np.sqrt(np.sum((_palette_lab - lab) ** 2, axis=1))
    idx = int(np.argmin(deltas))
    return {
        "name": _palette_names[idx],
        "hex": _palette_hexes[idx],
        "parent": _palette_parents[idx],
    }


# ---------------- Image extractors (unchanged) ----------------
def extract_garment_color(img_path_or_bytes):
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


# ---------------- Image hashing ----------------
def hash_image_bytes(img_bytes: bytes) -> str:
    """Stable hash for image-hash override lookups."""
    return hashlib.sha256(img_bytes).hexdigest()


# ---------------- Public API ----------------
def process_image(img_source, mode="garment", image_bytes: Optional[bytes] = None):
    """Extract and classify color from one image.

    If image_bytes is provided, we check the corrections store for an
    exact-image override before running the classifier.

    Returns dict:
        name, hex, parent      -- classified color
        sampled_rgb, sampled_hex -- the raw color we read from the image
        from_correction        -- True if served from image-hash override
        image_hash             -- sha256 of the image (so the frontend can
                                  reference it when submitting a correction)
    """
    extractor = extract_footwear_color if mode == "footwear" else extract_garment_color
    rgb = extractor(img_source)

    image_hash = None
    from_correction = False
    classified = None

    if image_bytes is not None:
        image_hash = hash_image_bytes(image_bytes)
        with _lock:
            override = _corrections.get("image_overrides", {}).get(image_hash)
        if override:
            classified = {
                "name": override["name"],
                "hex": override["hex"],
                "parent": override["parent"],
            }
            from_correction = True

    if classified is None:
        classified = classify_color(rgb)

    return {
        "name": classified["name"],
        "hex": classified["hex"],
        "parent": classified["parent"],
        "sampled_rgb": list(rgb),
        "sampled_hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
        "from_correction": from_correction,
        "image_hash": image_hash,
    }


# ---------------- Corrections write-path ----------------
def _save_corrections_locked():
    """Write current _corrections dict to disk. Caller must hold _lock."""
    tmp = _CORRECTIONS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_corrections, f, indent=2)
    tmp.replace(_CORRECTIONS_PATH)


def add_correction(
    image_hash: Optional[str],
    sampled_rgb: list,
    corrected_hex: str,
    corrected_name: Optional[str] = None,
    corrected_family: Optional[str] = None,
):
    """Record a user correction.

    Stores both:
    - image_override (if image_hash provided): exact-match shortcut.
    - user_anchor: the sampled RGB -> corrected color mapping that
      augments the palette for future similar images.

    Returns the new entry that will be used for classification.
    """
    corrected_hex = corrected_hex.upper()
    if not corrected_hex.startswith("#"):
        corrected_hex = "#" + corrected_hex

    # Default name = "Custom #RRGGBB" when user picks a free hex without naming
    if not corrected_name:
        corrected_name = f"Custom {corrected_hex}"
    if not corrected_family:
        # Auto-classify the family by finding nearest base-palette parent
        # from the picked hex (gives sensible default like "Blue" / "Brown").
        picked_rgb = _hex_to_rgb(corrected_hex)
        picked_lab = rgb_to_lab(np.asarray(picked_rgb, dtype=float))
        # Use only base palette for family inference (don't bootstrap from anchors)
        base_rgbs = np.array([_hex_to_rgb(e["hex"]) for e in BASE_PALETTE], dtype=float)
        base_lab = rgb_to_lab(base_rgbs)
        deltas = np.sqrt(np.sum((base_lab - picked_lab) ** 2, axis=1))
        idx = int(np.argmin(deltas))
        corrected_family = BASE_PALETTE[idx]["parent"]

    ts = datetime.now(timezone.utc).isoformat()
    anchor = {
        "name": corrected_name,
        "hex": corrected_hex,
        "parent": corrected_family,
        "rgb": [int(c) for c in sampled_rgb],
        "weight": 1,
        "ts": ts,
    }

    with _lock:
        _corrections.setdefault("user_anchors", []).append(anchor)
        if image_hash:
            _corrections.setdefault("image_overrides", {})[image_hash] = {
                "name": corrected_name,
                "hex": corrected_hex,
                "parent": corrected_family,
                "sampled_rgb": [int(c) for c in sampled_rgb],
                "ts": ts,
            }
        _save_corrections_locked()
        # Rebuild palette index so the new anchor is live for the next request
        _rebuild_palette_index()

    return {
        "name": corrected_name,
        "hex": corrected_hex,
        "parent": corrected_family,
    }


def get_corrections_snapshot() -> dict:
    """Return a deep-ish copy of the corrections store."""
    with _lock:
        return json.loads(json.dumps(_corrections))


def get_corrections_stats() -> dict:
    """Summary of what's been learned."""
    with _lock:
        return {
            "image_overrides_count": len(_corrections.get("image_overrides", {})),
            "user_anchors_count": len(_corrections.get("user_anchors", [])),
            "base_palette_size": len(BASE_PALETTE),
            "total_palette_size": len(_palette_names),
        }


def import_corrections(payload: dict, mode: str = "merge") -> dict:
    """Import a corrections payload (e.g. from a previous export).

    mode='merge'   -> add to existing
    mode='replace' -> overwrite
    """
    with _lock:
        global _corrections
        if mode == "replace":
            _corrections = {
                "image_overrides": dict(payload.get("image_overrides", {})),
                "user_anchors": list(payload.get("user_anchors", [])),
            }
        else:
            for k, v in payload.get("image_overrides", {}).items():
                _corrections.setdefault("image_overrides", {})[k] = v
            for anchor in payload.get("user_anchors", []):
                _corrections.setdefault("user_anchors", []).append(anchor)
        _save_corrections_locked()
        _rebuild_palette_index()
        return {
            "image_overrides_count": len(_corrections.get("image_overrides", {})),
            "user_anchors_count": len(_corrections.get("user_anchors", [])),
        }


def reset_corrections() -> dict:
    """Wipe all corrections (for debugging / re-starting calibration)."""
    with _lock:
        global _corrections
        _corrections = _empty_corrections()
        _save_corrections_locked()
        _rebuild_palette_index()
    return {"ok": True}
