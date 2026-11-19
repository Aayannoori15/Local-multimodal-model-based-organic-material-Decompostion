# Organic Waste Decomposition Stage Detector
**Waste to Wealth to Wheels · Phase 1 prototype (photo-based)**

A photo goes to a **MobileNetV2 CNN**, which predicts 1 of 4 decomposition stages.
Then **Gemma 4 E2B** (local, via Ollama) looks at the photo and the CNN result,
draws on a composting knowledge base, and explains the current state and the next steps.

| Stage | Looks like | Phase |
|---|---|---|
| 1 Fresh Waste | Raw peels/scraps, original shapes and colours | Mesophilic start |
| 2 Early Decomposition | Softening, browning, mould, still identifiable | Heating up |
| 3 Active Decomposition | Dark, moist, mixed, some fragments left | Thermophilic → cooling |
| 4 Mature Compost | Dark, crumbly, soil-like, earthy | Curing done, ready to use |

---

## Tonight's run order (about 1.5 hours total)

### 0. Setup (10 min)
```bash
cd compost-ai
python -m venv venv
# Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
ollama pull gemma4:e2b          # skip if you already have it
```

### 1. Get images (10 min download + 20 min cleaning)
```bash
python download_images.py
```
**Then open each folder in `dataset/` and delete bad images** (logos, people, text,
diagrams, empty bins, obviously wrong stage). Aim for **40+ good images per folder**.
You can also drop in your own phone photos. Just put them in the right folder.

### 2. Train (10–25 min on CPU, 2–5 min on GPU)
```bash
python train.py
# short on time?  python train.py --epochs1 5 --epochs2 4
```
This creates `models/compost_cnn.pt` and the plots for your slides in `results/`:
`confusion_matrix.png`, `training_curves.png`, `classification_report.txt`, `metrics.json`.

### 3. Run the demo
```bash
python app.py
```
Your browser opens http://127.0.0.1:7860. Upload a photo and click **Analyse**.

---

## Demo tips
- **Before presenting:** run one analysis so Gemma is already loaded into memory. The first call is slow.
- If Gemma is slow on your laptop, untick **"Let Gemma also look at the photo"**. Text-only is faster.
- If Ollama crashes during the demo, the app still shows knowledge-base advice, so the demo won't die.
- Keep 4 good test photos (one per stage) that are **not** in `dataset/` on your desktop.

## Troubleshooting
| Problem | Fix |
|---|---|
| `assert base.classes == CLASSES` error | `dataset/` must contain exactly the 4 stage folders |
| Bing downloads very few images | Run it again, or add more queries in `download_images.py` |
| Ollama 404 / model not found | `ollama list` and set `OLLAMA_MODEL` in `config.py` to the exact name |
| Gemma prints its reasoning | Already disabled with `think: false`; update Ollama if it still appears |
| Low accuracy | Clean the dataset more; stages 2 and 3 are the hardest to tell apart. Mention this as a Phase 2 motivation |

## Files
```
config.py            classes, paths, Ollama model name
knowledge.py         composting knowledge base (grounds the LLM advice)
download_images.py   Bing image scraper + dedupe/cleanup
train.py             MobileNetV2 transfer learning, 2-phase training, metrics and plots
app.py               Gradio web app: CNN → Gemma 4 E2B insights
```

## Phase 2 (roadmap)
Own dataset from real compost bins, photographed daily. Add sensors
(temperature, moisture, MQ-135 gas, pH) via ESP32 and fuse them with the image features
for true multimodal stage detection.
