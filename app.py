import os
import sqlite3
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import torch
import requests
import torchvision.models as models
import torchvision.transforms as transforms

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "products.db")

app = Flask(__name__)
app.secret_key = "your-secret-key"

# ✅ ResNet18 model (lightweight)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def image_to_embedding(pil_img: Image.Image) -> np.ndarray:
    img = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(img).cpu().numpy()[0]
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec.astype(np.float32)

def fetch_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def load_db_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, category, image_url, embedding FROM products")
    rows = cur.fetchall()
    conn.close()
    items = []
    for name, cat, img_url, emb in rows:
        vec = np.frombuffer(emb, dtype=np.float32)
        items.append({"name": name, "category": cat, "image_url": img_url, "vec": vec})
    return items

def top_k_similar(query_vec, items, k=10):
    sims = [(float(np.dot(query_vec, it["vec"])), it) for it in items]
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:k]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    uploaded_file = request.files.get("image")
    url = request.form.get("url", "").strip()
    img = None

    try:
        if uploaded_file and uploaded_file.filename != "":
            img = Image.open(uploaded_file.stream).convert("RGB")
        elif url:
            img = fetch_image_from_url(url)
        else:
            flash("Please upload an image or enter an image URL", "error")
            return redirect(url_for("index"))
    except Exception as e:
        flash(f"Error loading image: {e}", "error")
        return redirect(url_for("index"))

    query_vec = image_to_embedding(img)
    items = load_db_embeddings()
    if not items:
        flash("⚠️ Database is empty. Did you run prepare_data.py?", "error")
        return redirect(url_for("index"))

    top = top_k_similar(query_vec, items, k=10)
    results = [{"name": it["name"], "category": it["category"], "image_url": it["image_url"], "score": round(sim, 3)} for sim, it in top]

    return render_template("results.html", query_img=url if url else None, results=results)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
