"""
color_extractor.py

Color extraction + classification with multi-color detection.

Pipeline:
  1. Sample dominant region pixels.
  2. Filter out background (near-white, neutral gray) and skin (footwear).
  3. Cluster remaining pixels with k-means (k=3 in Lab space).
  4. Drop clusters under MIN_SHARE_PCT (noise/shadow/highlight).
  5. Merge clusters that are perceptually too close (Delta E < MERGE_THRESHOLD).
  6. Classify each remaining cluster against the augmented palette.
  7. Return list of color records sorted by share. is_multi=True iff 3+ distinct.

Calibration / corrections:
  - Each correction is per-color (the user can correct just the base, or just the
    print). The store keys corrections by the cluster's sampled RGB.
  - image_hash override is deprecated for multi-color (would need to be per-role)
    but is still set when the result is single-color, for backwards-compatible
    single-image pinning.

Public API (used by main.py):
  process_image(img_source, mode, image_bytes=None) -> {
      colors: [
        {role: 'base'|'print'|'print2', name, hex, parent, sampled_rgb,
         sampled_hex, share, from_correction},
        ...
      ],
      is_multi: bool,
      image_hash: str | None,
  }
  add_correction(...)
  get_corrections_snapshot()
  get_corrections_stats()
  import_corrections(...)
  reset_corrections()
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

# ---------------- Tunable thresholds ----------------
MAX_CLUSTERS = 3                # k for k-means
MIN_SHARE_PCT = 0.04            # cluster must be >=4% of sampled pixels (catches sparse prints like florals)
MERGE_THRESHOLD = 28.0          # Lab Delta E below which clusters are considered the same color (merges shadows/highlights into base)
KMEANS_ITERS = 12               # plenty for 3 centroids on a few thousand pixels
KMEANS_SAMPLE_LIMIT = 5000      # cap pixels fed into k-means for speed
ANCHOR_INFLUENCE_THRESHOLD = 30.0  # Lab Delta E within which a user anchor wins over base palette


# ---------------- Paths ----------------
_API_DIR = Path(__file__).parent
_PALETTE_PATH = _API_DIR / "palette.json"

_CORRECTIONS_DIR_ENV = os.environ.get("CHROMATIC_DATA_DIR")
_DATA_DIR = Path(_CORRECTIONS_DIR_ENV) if _CORRECTIONS_DIR_ENV else (_API_DIR / "data")
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_CORRECTIONS_PATH = _DATA_DIR / "corrections.json"


# ---------------- Palette + corrections loading ----------------
with open(_PALETTE_PATH, "r", encoding="utf-8") as _f:
    BASE_PALETTE = json.load(_f)


def _hex_to_rgb(hex_str):
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb):
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _empty_corrections():
    return {"image_overrides": {}, "user_anchors": []}


def _load_corrections():
    if not _CORRECTIONS_PATH.exists():
        return _empty_corrections()
    try:
        with open(_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("image_overrides", {})
        data.setdefault("user_anchors", [])
        return data
    except Exception:
        return _empty_corrections()


_lock = threading.Lock()
_corrections = _load_corrections()


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


# ---------------- Augmented palette index ----------------
_palette_names: list = []
_palette_hexes: list = []
_palette_parents: list = []
_palette_lab: np.ndarray = np.zeros((0, 3))


def _rebuild_palette_index():
    global _palette_names, _palette_hexes, _palette_parents, _palette_lab

    entries = []
    for e in BASE_PALETTE:
        rgb = _hex_to_rgb(e["hex"])
        entries.append({
            "name": e["name"],
            "hex": e["hex"],
            "parent": e["parent"],
            "rgb": rgb,
        })

    # User anchors are only added if their corrected color isn't already a near-duplicate
    # of an existing entry — keeps the palette from drifting on near-misses.
    base_lab = rgb_to_lab(np.array([e["rgb"] for e in entries], dtype=float))
    for anchor in _corrections.get("user_anchors", []):
        anchor_rgb = tuple(anchor.get("corrected_rgb") or _hex_to_rgb(anchor["hex"]))
        anchor_lab = rgb_to_lab(np.asarray(anchor_rgb, dtype=float))
        deltas = np.sqrt(np.sum((base_lab - anchor_lab) ** 2, axis=1))
        nearest = int(np.argmin(deltas))
        if deltas[nearest] < 8.0 and entries[nearest]["name"].lower() == anchor["name"].lower():
            # Same name and very close — already represented, skip
            continue
        entries.append({
            "name": anchor["name"],
            "hex": anchor["hex"],
            "parent": anchor["parent"],
            "rgb": anchor_rgb,
        })

    _palette_names = [e["name"] for e in entries]
    _palette_hexes = [e["hex"] for e in entries]
    _palette_parents = [e["parent"] for e in entries]
    rgbs = np.array([e["rgb"] for e in entries], dtype=float)
    _palette_lab = rgb_to_lab(rgbs)


_rebuild_palette_index()


# ---------------- Single-RGB classification ----------------
def classify_color(rgb):
    """Map a single RGB to the nearest palette entry. Returns {name, hex, parent}."""
    lab = rgb_to_lab(np.asarray(rgb, dtype=float))
    deltas = np.sqrt(np.sum((_palette_lab - lab) ** 2, axis=1))
    idx = int(np.argmin(deltas))
    return {
        "name": _palette_names[idx],
        "hex": _palette_hexes[idx],
        "parent": _palette_parents[idx],
    }


def classify_with_anchor_check(rgb):
    """Like classify_color, but also returns whether a user anchor was the winner.

    User anchors are appended to the augmented palette. We can detect that the
    chosen entry came from a correction by checking its index against the base
    palette length.
    """
    lab = rgb_to_lab(np.asarray(rgb, dtype=float))
    deltas = np.sqrt(np.sum((_palette_lab - lab) ** 2, axis=1))
    idx = int(np.argmin(deltas))
    return {
        "name": _palette_names[idx],
        "hex": _palette_hexes[idx],
        "parent": _palette_parents[idx],
    }, idx >= len(BASE_PALETTE)


# ---------------- Pixel sampling (per mode) ----------------
def _sample_garment_pixels(arr):
    h, w, _ = arr.shape
    regions = [
        arr[int(h * 0.30): int(h * 0.45), int(w * 0.38): int(w * 0.62)],
        arr[int(h * 0.45): int(h * 0.60), int(w * 0.35): int(w * 0.65)],
        arr[int(h * 0.55): int(h * 0.68), int(w * 0.38): int(w * 0.62)],
    ]
    return np.vstack([r.reshape(-1, 3) for r in regions])


def _sample_footwear_pixels(arr):
    h, w, _ = arr.shape
    regions = [
        arr[int(h * 0.35): int(h * 0.55), int(w * 0.20): int(w * 0.50)],
        arr[int(h * 0.35): int(h * 0.55), int(w * 0.50): int(w * 0.80)],
        arr[int(h * 0.55): int(h * 0.80), int(w * 0.20): int(w * 0.50)],
        arr[int(h * 0.55): int(h * 0.80), int(w * 0.50): int(w * 0.80)],
        arr[int(h * 0.45): int(h * 0.75), int(w * 0.35): int(w * 0.65)],
    ]
    return np.vstack([r.reshape(-1, 3) for r in regions])


def _filter_pixels(all_px, mode):
    """Strip background and (for footwear) skin from the sampled pixels."""
    r_ch = all_px[:, 0].astype(int)
    g_ch = all_px[:, 1].astype(int)
    b_ch = all_px[:, 2].astype(int)

    pre_median = np.median(all_px, axis=0)
    pm_min = pre_median.min()
    pm_max = pre_median.max()
    product_is_white = all(c > 215 for c in pre_median)
    product_is_gray = (pm_max - pm_min) < 8 and 180 < pm_min < 225

    kill_masks = []

    # Near-white background — strip unless the product itself is near-white
    if not product_is_white:
        kill_masks.append((r_ch > 228) & (g_ch > 228) & (b_ch > 228))

    # Neutral gray background (concrete/walls) — strip unless product is gray
    if not product_is_gray and not product_is_white:
        neutral = (
            (abs(r_ch - g_ch) < 8) & (abs(g_ch - b_ch) < 8)
            & (abs(r_ch - b_ch) < 10) & (r_ch > 150) & (r_ch < 230)
        )
        kill_masks.append(neutral)

    if mode == "footwear":
        skin = (
            (r_ch > g_ch) & (g_ch > b_ch)
            & ((r_ch - b_ch) > 25) & ((r_ch - b_ch) < 100)
            & (r_ch > 130) & (r_ch < 245)
            & ((r_ch - g_ch) <= (r_ch - b_ch))
            & (abs((g_ch - b_ch) - (r_ch - g_ch)) < 30)
        )
        # Only strip skin if it isn't the dominant content (otherwise the product itself is skin-toned)
        if skin.sum() < len(all_px) * 0.65:
            kill_masks.append(skin)

    if not kill_masks:
        return all_px

    combined = np.zeros(len(all_px), dtype=bool)
    for m in kill_masks:
        combined |= m
    keep = ~combined
    if keep.sum() < 500:
        return all_px  # filtering would leave too little, abort
    return all_px[keep]


# ---------------- K-means in Lab space ----------------
def _kmeans_lab(pixels_rgb, k, iters=KMEANS_ITERS, seed=42):
    """Simple k-means on pixels in Lab space. Returns (labels, centroids_rgb, shares)."""
    n = len(pixels_rgb)
    if n == 0:
        return np.array([]), np.empty((0, 3)), np.array([])
    if n < k:
        k = max(1, n)

    # Subsample for speed
    rng = np.random.default_rng(seed)
    if n > KMEANS_SAMPLE_LIMIT:
        sample_idx = rng.choice(n, KMEANS_SAMPLE_LIMIT, replace=False)
        sample_rgb = pixels_rgb[sample_idx]
    else:
        sample_rgb = pixels_rgb

    sample_lab = rgb_to_lab(sample_rgb.astype(float))

    # k-means++ seeding for stability
    centroids = np.empty((k, 3))
    centroids[0] = sample_lab[rng.integers(0, len(sample_lab))]
    for i in range(1, k):
        d2 = np.min(np.sum((sample_lab[:, None, :] - centroids[:i][None, :, :]) ** 2, axis=2), axis=1)
        if d2.sum() == 0:
            centroids[i] = sample_lab[rng.integers(0, len(sample_lab))]
            continue
        probs = d2 / d2.sum()
        cum = np.cumsum(probs)
        r = rng.random()
        idx = int(np.searchsorted(cum, r))
        idx = min(idx, len(sample_lab) - 1)
        centroids[i] = sample_lab[idx]

    # Lloyd's iterations — vectorized
    for _ in range(iters):
        d = np.sqrt(np.sum((sample_lab[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
        labels = np.argmin(d, axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centroids[c] = sample_lab[mask].mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=0.5):
            break
        centroids = new_centroids

    # Final labels for the sample
    d = np.sqrt(np.sum((sample_lab[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    labels = np.argmin(d, axis=1)

    shares = np.array([(labels == c).sum() / len(labels) for c in range(k)])

    # Centroid RGB = mean of original RGBs in each cluster
    centroids_rgb = np.empty((k, 3))
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids_rgb[c] = sample_rgb[mask].mean(axis=0)
        else:
            centroids_rgb[c] = np.array([128, 128, 128])

    return labels, centroids_rgb, shares


# ---------------- Cluster post-processing ----------------
def _merge_close_clusters(centroids_rgb, shares):
    """Merge clusters whose Lab Delta E is below MERGE_THRESHOLD.

    Returns (merged_centroids_rgb, merged_shares).
    """
    if len(centroids_rgb) <= 1:
        return centroids_rgb, shares

    centroids_lab = rgb_to_lab(centroids_rgb.astype(float))
    n = len(centroids_lab)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centroids_lab[i] - centroids_lab[j]) ** 2))
            if d < MERGE_THRESHOLD:
                union(i, j)

    # Group
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged_rgb = []
    merged_shares = []
    for members in groups.values():
        total_share = sum(shares[m] for m in members)
        # Weighted-by-share centroid in RGB
        weighted = np.sum([centroids_rgb[m] * shares[m] for m in members], axis=0) / total_share
        merged_rgb.append(weighted)
        merged_shares.append(total_share)

    merged_rgb = np.array(merged_rgb)
    merged_shares = np.array(merged_shares)
    # Sort by share descending
    order = np.argsort(-merged_shares)
    return merged_rgb[order], merged_shares[order]


# ---------------- High-level extraction ----------------
def extract_colors(img_path_or_bytes, mode):
    """Extract one or more dominant colors from an image.

    Returns list of {centroid_rgb, share}, sorted by share descending,
    with at most 3 entries and each entry having share >= MIN_SHARE_PCT.
    """
    img = Image.open(img_path_or_bytes).convert("RGB")
    arr = np.array(img)

    if mode == "footwear":
        all_px = _sample_footwear_pixels(arr)
    else:
        all_px = _sample_garment_pixels(arr)

    filtered = _filter_pixels(all_px, mode)

    if len(filtered) == 0:
        return [{"centroid_rgb": (128, 128, 128), "share": 1.0}]

    # k-means on filtered
    _labels, centroids_rgb, shares = _kmeans_lab(filtered, k=MAX_CLUSTERS)

    # Drop tiny clusters before merging
    keep = shares >= MIN_SHARE_PCT
    if keep.sum() == 0:
        # Fallback: keep the largest
        idx = int(np.argmax(shares))
        centroids_rgb = centroids_rgb[idx:idx + 1]
        shares = shares[idx:idx + 1]
    else:
        centroids_rgb = centroids_rgb[keep]
        shares = shares[keep]

    # Merge perceptually-close clusters (e.g. shadowed-black + lit-black)
    centroids_rgb, shares = _merge_close_clusters(centroids_rgb, shares)

    # Renormalize shares
    shares = shares / shares.sum()

    # Re-sort by share desc
    order = np.argsort(-shares)
    centroids_rgb = centroids_rgb[order]
    shares = shares[order]

    return [
        {"centroid_rgb": tuple(int(c) for c in centroids_rgb[i]), "share": float(shares[i])}
        for i in range(len(centroids_rgb))
    ]


# ---------------- Image hashing ----------------
def hash_image_bytes(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()


# ---------------- Public top-level API ----------------
ROLES = ["base", "print", "print2"]


def process_image(img_source, mode="garment", image_bytes: Optional[bytes] = None):
    """Extract & classify colors. Returns dict:
        {
          colors: [
              {role, name, hex, parent, sampled_rgb, sampled_hex,
               share, from_correction},
              ...
          ],
          is_multi: bool,            # True iff len(colors) >= 3 distinct
          image_hash: str | None,
        }
    """
    image_hash = hash_image_bytes(image_bytes) if image_bytes else None

    # Image-hash override (only applies when we have an exact-image pin)
    override = None
    if image_hash:
        with _lock:
            override = _corrections.get("image_overrides", {}).get(image_hash)

    if override:
        # Reconstruct response from saved override (preserves color list)
        return {
            "colors": [
                {
                    "role": c.get("role", ROLES[i] if i < len(ROLES) else f"color{i}"),
                    "name": c["name"],
                    "hex": c["hex"],
                    "parent": c["parent"],
                    "sampled_rgb": c.get("sampled_rgb", _hex_to_rgb(c["hex"])),
                    "sampled_hex": _rgb_to_hex(c.get("sampled_rgb", _hex_to_rgb(c["hex"]))),
                    "share": c.get("share", 1.0 / len(override["colors"])),
                    "from_correction": True,
                }
                for i, c in enumerate(override["colors"])
            ],
            "is_multi": override.get("is_multi", False),
            "image_hash": image_hash,
        }

    # Standard path
    extracted = extract_colors(img_source, mode=mode)
    is_multi = len(extracted) >= 3

    colors = []
    for i, item in enumerate(extracted):
        sampled_rgb = item["centroid_rgb"]
        share = item["share"]
        classified, from_anchor = classify_with_anchor_check(sampled_rgb)
        colors.append({
            "role": ROLES[i] if i < len(ROLES) else f"color{i}",
            "name": classified["name"],
            "hex": classified["hex"],
            "parent": classified["parent"],
            "sampled_rgb": list(sampled_rgb),
            "sampled_hex": _rgb_to_hex(sampled_rgb),
            "share": share,
            "from_correction": from_anchor,
        })

    return {
        "colors": colors,
        "is_multi": is_multi,
        "image_hash": image_hash,
    }


# ---------------- Corrections write-path ----------------
def _save_corrections_locked():
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
    role: Optional[str] = None,
    full_colors_for_pin: Optional[list] = None,
):
    """Record a user correction for one color (one role on one image).

    Stores:
      - user_anchor: (sampled_rgb -> corrected name/hex/parent) so future
        images with similar dominant colors classify correctly.
      - image_override (if image_hash + full_colors_for_pin provided):
        a complete pinned answer for that exact image (all roles).
        full_colors_for_pin is the list of *all* colors for the image
        with the corrected one substituted, so re-uploading the same
        image returns the full corrected multi-color answer.

    Returns the saved color.
    """
    corrected_hex = corrected_hex.upper()
    if not corrected_hex.startswith("#"):
        corrected_hex = "#" + corrected_hex

    if not corrected_name:
        corrected_name = f"Custom {corrected_hex}"
    if not corrected_family:
        # Auto-detect family from the picked hex against base palette only
        picked_rgb = _hex_to_rgb(corrected_hex)
        picked_lab = rgb_to_lab(np.asarray(picked_rgb, dtype=float))
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
        "rgb": [int(c) for c in sampled_rgb],   # the rgb we extracted from the image
        "corrected_rgb": list(_hex_to_rgb(corrected_hex)),
        "role": role,
        "ts": ts,
    }

    with _lock:
        _corrections.setdefault("user_anchors", []).append(anchor)
        if image_hash and full_colors_for_pin:
            _corrections.setdefault("image_overrides", {})[image_hash] = {
                "colors": full_colors_for_pin,
                "is_multi": len(full_colors_for_pin) >= 3,
                "ts": ts,
            }
        _save_corrections_locked()
        _rebuild_palette_index()

    return {
        "name": corrected_name,
        "hex": corrected_hex,
        "parent": corrected_family,
    }


def get_corrections_snapshot() -> dict:
    with _lock:
        return json.loads(json.dumps(_corrections))


def get_corrections_stats() -> dict:
    with _lock:
        return {
            "image_overrides_count": len(_corrections.get("image_overrides", {})),
            "user_anchors_count": len(_corrections.get("user_anchors", [])),
            "base_palette_size": len(BASE_PALETTE),
            "total_palette_size": len(_palette_names),
        }


def import_corrections(payload: dict, mode: str = "merge") -> dict:
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
    with _lock:
        global _corrections
        _corrections = _empty_corrections()
        _save_corrections_locked()
        _rebuild_palette_index()
    return {"ok": True}
