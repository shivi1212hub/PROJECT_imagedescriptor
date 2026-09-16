"""
build_index.py
----------------
One-time preprocessing script.

Scans a folder of photos, encodes each image into a CLIP embedding,
and stores all embeddings in a FAISS index on disk (along with a
mapping back to the original filenames).

Usage:
    python build_index.py --photos_dir ./photos --output_dir ./index
"""

import argparse
import json
import os

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(photos_dir: str) -> list[str]:
    """Return a sorted list of image file paths inside photos_dir."""
    paths = []
    for root, _, files in os.walk(photos_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def build_index(photos_dir: str, output_dir: str, model_name: str = "clip-ViT-B-32", batch_size: int = 32):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = find_images(photos_dir)
    if not image_paths:
        raise SystemExit(f"No supported images found in {photos_dir}")
    print(f"Found {len(image_paths)} images.")

    print(f"Loading CLIP model '{model_name}' (downloads automatically on first run)...")
    model = SentenceTransformer(model_name)

    embeddings = []
    valid_paths = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
                batch_valid_paths.append(p)
            except Exception as e:
                print(f"  Skipping {p}: {e}")

        if not batch_images:
            continue

        batch_embeddings = model.encode(
            batch_images,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # so we can use inner product == cosine similarity
        )

        embeddings.append(batch_embeddings)
        valid_paths.extend(batch_valid_paths)
        print(f"  Encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    embeddings = np.vstack(embeddings).astype("float32")

    # Inner product on normalized vectors == cosine similarity search
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(output_dir, "photos.index"))

    with open(os.path.join(output_dir, "filenames.json"), "w") as f:
        json.dump(valid_paths, f)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"model_name": model_name, "dim": dim, "count": len(valid_paths)}, f)

    print(f"\nDone. Indexed {len(valid_paths)} images.")
    print(f"Saved to: {output_dir}/photos.index and {output_dir}/filenames.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a CLIP + FAISS index over a photo library.")
    parser.add_argument("--photos_dir", required=True, help="Folder containing your photos (searched recursively).")
    parser.add_argument("--output_dir", default="./index", help="Where to save the FAISS index and metadata.")
    parser.add_argument("--model_name", default="clip-ViT-B-32", help="SentenceTransformers CLIP model to use.")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    build_index(args.photos_dir, args.output_dir, args.model_name, args.batch_size)
