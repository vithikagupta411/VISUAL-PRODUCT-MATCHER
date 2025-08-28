"""
Prepare the products database using ResNet18 embeddings.
"""

import os
import csv
import sqlite3
from io import BytesIO
import numpy as np
import requests
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "products.csv")
DB_PATH = os.path.join(APP_DIR, "products.db")

# ✅ ResNet18 model (lightweight)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # remove classification layer → get features
model = model.to(device)
model.eval()

# Image preprocessing
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
    vec = vec / (np.linalg.norm(vec) + 1e-10)  # normalize
    return vec.astype(np.float32)

def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            image_url TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()

def insert_product(conn, name, category, image_url, vec):
    cur = conn.cursor()
    cur.execute("SELECT id FROM products WHERE image_url = ?", (image_url,))
    if cur.fetchone():
        print(f"Skipping duplicate: {name}")
        return
    cur.execute(
        "INSERT INTO products (name, category, image_url, embedding) VALUES (?, ?, ?, ?)",
        (name, category, image_url, vec.tobytes())
    )
    conn.commit()

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Create it with columns: name,category,image_url")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        raise ValueError("products.csv is empty. Add at least 50 rows.")
    print(f"Found {len(rows)} products. Processing...")

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    for i, row in enumerate(rows, 1):
        name, category, image_url = row["name"].strip(), row["category"].strip(), row["image_url"].strip()
        if not (name and category and image_url):
            print(f"[skip row {i}] Missing fields")
            continue
        try:
            img = fetch_image(image_url)
            vec = image_to_embedding(img)
            insert_product(conn, name, category, image_url, vec)
            print(f"[{i}/{len(rows)}] Added: {name}")
        except Exception as e:
            print(f"[error row {i}] {name} -> {e}")

    conn.close()
    print("✅ Done. products.db created/updated.")

if __name__ == "__main__":
    main()
