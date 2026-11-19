"""Step 3: Run the prototype web app.

    ollama serve                 # (if Ollama is not already running)
    python app.py                # opens http://127.0.0.1:7860

Pipeline: photo -> MobileNetV2 CNN (stage + confidence)
                -> Gemma 4 E2B via Ollama (sees the photo + CNN result + knowledge base)
                -> current state, what is happening, what to do next.
If Ollama is not reachable the app still works and shows knowledge-base advice.
"""

import base64
import io
import json

import gradio as gr
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import (CLASS_LABELS, CLASSES, DATA_DIR, IMG_SIZE, MODEL_DIR,
                    OLLAMA_MODEL, OLLAMA_URL)
from knowledge import STAGE_KNOWLEDGE, stage_context

# ---------------- CNN ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(MODEL_DIR / "compost_cnn.pt", map_location=device)
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, len(CLASSES)))
model.load_state_dict(ckpt["state_dict"])
model.eval().to(device)

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def classify(img: Image.Image):
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0].cpu().tolist()
    return {CLASSES[i]: p for i, p in enumerate(probs)}


# ---------------- LLM ----------------
SYSTEM = (
    "You are a composting expert assistant inside an organic-waste monitoring system "
    "for a 'Waste to Wealth' project. Be practical, specific and concise. "
    "Base advice on the provided knowledge base. Use simple language for households and farmers."
)


def build_prompt(probs, use_vision):
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, conf = ranked[0]
    second, conf2 = ranked[1]
    prob_text = "\n".join(f"- {CLASS_LABELS[c]}: {p:.1%}" for c, p in ranked)
    uncertain = conf < 0.55
    p = (
        f"A CNN image classifier analysed a photo of organic waste.\n"
        f"Predicted: {CLASS_LABELS[top]} (confidence {conf:.1%})\n"
        f"All stage probabilities:\n{prob_text}\n\n"
        f"Knowledge base for the predicted stage:\n{stage_context(top)}\n\n"
    )
    if uncertain:
        p += (f"The CNN is uncertain between {CLASS_LABELS[top]} and {CLASS_LABELS[second]} "
              f"({conf2:.1%}). Knowledge for the second option:\n{stage_context(second)}\n\n")
    if use_vision:
        p += ("You can also see the photo. Look at colour, texture, visible food pieces, mould and moisture.\n\n")
    p += "Reply in markdown with exactly these sections:\n"
    p += "### Current state\n(2-3 sentences: which stage and what it looks like)\n"
    p += "### What is happening inside\n(1-2 sentences on the microbial/decomposition process)\n"
    p += "### What to do next\n(3-4 short bullet points, specific actions)\n"
    p += "### Time to usable compost\n(one line estimate)\n"
    if use_vision:
        p += ("### Visual cross-check\n(Do you agree with the CNN's stage based on what you see in the photo? "
              "Answer 'Agree' or 'Disagree - looks like Stage N' with a one-line reason.)\n")
    return p


def img_b64(img: Image.Image):
    img = img.convert("RGB")
    img.thumbnail((768, 768))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def stream_ollama(prompt, image=None):
    msg = {"role": "user", "content": prompt}
    if image is not None:
        msg["images"] = [img_b64(image)]
    body = {"model": OLLAMA_MODEL, "stream": True, "think": False,
            "messages": [{"role": "system", "content": SYSTEM}, msg],
            "options": {"temperature": 0.3}}
    r = requests.post(OLLAMA_URL, json=body, stream=True, timeout=300)
    if r.status_code == 400:  # older Ollama without the 'think' option
        body.pop("think")
        r = requests.post(OLLAMA_URL, json=body, stream=True, timeout=300)
    r.raise_for_status()
    text = ""
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        text += chunk.get("message", {}).get("content", "")
        yield text
        if chunk.get("done"):
            break


def fallback_advice(cls):
    k = STAGE_KNOWLEDGE[cls]
    acts = "\n".join(f"- {a}" for a in k["actions"])
    return (f"*Ollama not reachable, so this is knowledge-base advice.*\n\n"
            f"### Current state\n{k['summary']}\n\n### What is happening inside\n{k['process']}\n\n"
            f"### What to do next\n{acts}\n\n### Time to usable compost\n{k['next_stage_eta']}\n\n"
            f"**Note:** {k['warnings']}")


# ---------------- UI ----------------
STAGE_COLORS = {"1_fresh_waste": "#7cb342", "2_early_decomposition": "#c0a030",
                "3_active_decomposition": "#8d5a2b", "4_mature_compost": "#3e2723"}


def stage_card(probs):
    top = max(probs, key=probs.get)
    conf = probs[top]
    idx = CLASSES.index(top) + 1
    bar = "".join(
        f"<div style='flex:1;height:12px;border-radius:6px;background:{STAGE_COLORS[c] if i < idx else '#ddd'}'></div>"
        for i, c in enumerate(CLASSES))
    note = "" if conf >= 0.55 else "<div style='color:#c62828;margin-top:6px'>Low confidence: borderline between stages</div>"
    return (f"<div style='padding:14px;border-radius:12px;border:2px solid {STAGE_COLORS[top]}'>"
            f"<div style='font-size:22px;font-weight:700'>{CLASS_LABELS[top]}</div>"
            f"<div style='opacity:.75'>Confidence {conf:.1%} · {STAGE_KNOWLEDGE[top]['typical_days']}</div>"
            f"<div style='display:flex;gap:6px;margin-top:10px'>{bar}</div>{note}</div>")


def analyse(image, use_vision):
    if image is None:
        yield {}, "", "Upload a photo first."
        return
    probs = classify(image)
    label = {CLASS_LABELS[c]: p for c, p in probs.items()}
    card = stage_card(probs)
    yield label, card, f"*Asking {OLLAMA_MODEL} for insights...*"
    try:
        for text in stream_ollama(build_prompt(probs, use_vision), image if use_vision else None):
            yield label, card, text
    except Exception as e:
        print("Ollama error:", e)
        yield label, card, fallback_advice(max(probs, key=probs.get))


examples = []
for c in CLASSES:
    files = sorted((DATA_DIR / c).glob("*.jpg"))[:1] if (DATA_DIR / c).exists() else []
    examples += [[str(f), True] for f in files]

THEME = gr.themes.Soft(primary_hue="green")
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6  # Gradio 6 moved `theme` to launch()
blocks_kwargs = {} if GRADIO_6 else {"theme": THEME}

with gr.Blocks(title="Organic Waste Decomposition Detector", **blocks_kwargs) as demo:
    gr.Markdown("# ♻️ Organic Waste Decomposition Stage Detector\n"
                "**Waste → Wealth:** CNN (MobileNetV2) detects the decomposition stage, "
                f"local LLM (**{OLLAMA_MODEL}** via Ollama) explains it and recommends next steps.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Photo of organic waste / compost", height=320)
            vis = gr.Checkbox(value=True, label="Let Gemma also look at the photo (visual cross-check)")
            btn = gr.Button("Analyse", variant="primary")
            if examples:
                gr.Examples(examples, [inp, vis], label="Sample images")
        with gr.Column(scale=1):
            card = gr.HTML()
            lab = gr.Label(num_top_classes=4, label="CNN stage probabilities")
        with gr.Column(scale=1):
            out = gr.Markdown(label="AI insights")
    btn.click(analyse, [inp, vis], [lab, card, out])

if __name__ == "__main__":
    demo.launch(inbrowser=True, **({"theme": THEME} if GRADIO_6 else {}))
