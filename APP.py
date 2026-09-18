"""
app.py
------
Gradio web app: type a text description, get back the most visually
and semantically similar photos from your indexed library.

Run this AFTER build_index.py has created ./index/photos.index

Usage:
    python app.py --index_dir ./index
"""

import argparse
import json
import os

import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

# ---- Globals populated at startup ----
model = None
index = None
filenames = None


def load_resources(index_dir: str):
    global model, index, filenames

    with open(os.path.join(index_dir, "config.json")) as f:
        config = json.load(f)

    with open(os.path.join(index_dir, "filenames.json")) as f:
        filenames = json.load(f)

    print(f"Loading model '{config['model_name']}'...")
    model = SentenceTransformer(config["model_name"])

    print("Loading FAISS index...")
    index = faiss.read_index(os.path.join(index_dir, "photos.index"))

    print(f"Ready. {config['count']} photos indexed.")


def search(query: str, top_k: int = 10):
    if not query.strip():
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        path = filenames[idx]
        caption = f"{os.path.basename(path)}  (score: {score:.3f})"
        results.append((path, caption))

    return results


def build_ui():
    with gr.Blocks(title="Photo Search (CLIP + FAISS)") as demo:
        gr.Markdown(
            """
            # 🔍 Search your photos by description
            Type what you're looking for (e.g. *"a dog running on the beach"*,
            *"sunset over mountains"*, *"birthday cake with candles"*) and the
            most semantically similar photos will show up below.
            """
        )
        with gr.Row():
            query_box = gr.Textbox(
                label="Describe the photo you're looking for",
                placeholder="e.g. a red car parked in front of a building",
                scale=4,
            )
            top_k_slider = gr.Slider(1, 30, value=10, step=1, label="Number of results", scale=1)

        search_button = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(label="Results", columns=5, height="auto")

        search_button.click(fn=search, inputs=[query_box, top_k_slider], outputs=gallery)
        query_box.submit(fn=search, inputs=[query_box, top_k_slider], outputs=gallery)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the CLIP photo search app.")
    parser.add_argument("--index_dir", default="./index", help="Folder containing photos.index / filenames.json")
    parser.add_argument("--share", action="store_true", help="Create a public shareable Gradio link.")
    args = parser.parse_args()

    load_resources(args.index_dir)
    ui = build_ui()
    ui.launch(share=args.share)
