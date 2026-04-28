# Chromatic — Garment Color Extractor

A self-hosted web app that takes a zip of product images (named by SKU) and returns an xlsx file with the **canonical color name**, **hex code**, and **parent color** for each item.

The classifier uses a curated palette of ~285 brand-canonical colors (e.g. Chocolate, Burgundy, Dusty Rose, Midnight Navy, Charcoal Black, Soft Powder Pink, Raspberry Pink, Sage Green, etc.) and matches sampled image colors using **Delta E in CIELAB color space** for perceptually accurate naming.

Output columns: `SKU | Color | Color Code | Parent Color`

Two extraction modes:
- **Garment** — for skirts, dresses, tops, co-ords. Samples the center of the frame.
- **Footwear** — for shoes, boots, sandals. Handles skin (feet/legs) and busy lifestyle backgrounds.

---

## Quick Start (run locally)

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
cd api
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

---

## Project Structure

```
color_app/
├── api/
│   ├── main.py               # FastAPI backend
│   └── color_extractor.py    # Core color extraction + classification
├── static/
│   └── index.html            # Frontend UI
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment config
├── vercel.json               # Vercel deployment config
├── Procfile                  # Railway/Heroku-style config
├── runtime.txt               # Python version pin
└── README.md
```

---

## Deployment (Free Hosting)

### Option A — Render (recommended, no size/time limits for this workload)

Render's free web service tier handles zips of 100+ images cleanly. Spins down after 15 min of inactivity, spins back up in ~30 seconds on the next request.

**Steps:**

1. Push this project to a GitHub repo (see "GitHub via Antigravity" below).
2. Go to https://render.com and sign in with GitHub.
3. Click **New → Web Service** and connect your repo.
4. Render will auto-detect `render.yaml`. Confirm the settings and click **Create Web Service**.
5. Wait ~3–4 minutes for first build. You'll get a public URL like `https://chromatic-color-extractor.onrender.com`.

### Option B — Vercel (works for small zips only)

Vercel's free tier runs Python as serverless functions. The limits:
- **10 seconds** max execution per request (so zips > ~40 images may time out)
- **4.5 MB** max request body

Good for quick tests, not great for large batches.

**Steps:**

1. Push this project to GitHub.
2. Go to https://vercel.com → **Add New → Project** → import your repo.
3. Framework preset: **Other**. Leave build/output blank.
4. Click **Deploy**. You'll get a URL like `https://your-project.vercel.app`.

### Option C — Railway (generous free trial; paid after)

Railway gives $5/month free credit (enough for a low-traffic app). Uses the `Procfile`.

1. https://railway.app → **New Project → Deploy from GitHub** → pick the repo.
2. Railway auto-detects the Procfile. Click **Deploy**.

---

## Pushing to GitHub via Antigravity

1. Open the `color_app` folder in Antigravity (or your IDE).
2. Initialize git and commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Chromatic color extractor"
   ```
3. Create a new repo on GitHub (https://github.com/new), then:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/chromatic.git
   git push -u origin main
   ```
4. Connect the GitHub repo to Render or Vercel as described above.

---

## API Endpoint

If you want to call the backend directly (e.g. from a script):

```bash
curl -X POST https://your-deployed-url.com/extract \
  -F "file=@your_zip.zip" \
  -F "mode=garment" \
  -o output.xlsx
```

Returns an `.xlsx` with four columns: `SKU`, `Color`, `Color Code`, `Parent Color`.

Response headers include:
- `X-Processed-Count` — number of images processed
- `X-Error-Count` — number that failed

---

## How the Classifier Works

1. **Sampling** — grabs multiple regions of the image (center-vertical for garments; lower-center for footwear).
2. **Background removal** — strips near-white and neutral-gray pixels (product backgrounds).
3. **Skin removal** (footwear only) — filters out feet/leg tones.
4. **Busy-scene detection** (footwear only) — if the sample has wide brightness/hue variance, it assumes a lifestyle shot and picks the dominant dark cluster (usually the shoe).
5. **Median color** — takes the median of the filtered pixels (robust to outliers).
6. **Classification** — converts the sampled RGB to CIELAB color space, then finds the perceptually nearest entry in the canonical palette (`api/palette.json`) using Delta E (CIE76) distance. Returns the canonical color name, its established hex code, and parent color group.

### Customizing the Palette

The color palette is in `api/palette.json` — a list of objects with `name`, `hex`, and `parent`. Add, remove, or edit entries to tune classifications. No code changes needed.

### Known Limitations

- **Patterned prints** (leopard, floral, lemon print) — returns the dominant color, which may be the base OR the print depending on proportions.
- **Lifestyle shots with busy scenery** (grass fields, bright chairs, colorful co-models) may bleed into the sample. For best accuracy, use clean studio shots.
- **Multi-color co-ords** where top and bottom differ — returns the color of the center of the frame.

---

## Supported Image Formats

- `.webp`
- `.jpg` / `.jpeg`
- `.png`

Image filenames (without extension) are used as the SKU in the output xlsx. Zip folder structure doesn't matter — all images are flattened.

---

## License

MIT — do whatever you want.
