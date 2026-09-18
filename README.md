# Photo Search — Search Photos Using Text

A Python application that lets you search through a collection of photos using natural language instead of filenames.

Instead of remembering `IMG_2048.jpg`, you can simply type:

> `a dog running on the beach`

The project uses **CLIP** to understand both images and text, **FAISS** to perform fast similarity search, and **Gradio** to provide a simple web interface.

---

## How It Works

The project has two main stages:

1. **Build the image index**
2. **Search the indexed images**

During indexing:

* Every image is converted into a **CLIP embedding**.
* The embeddings are stored inside a **FAISS index**.
* The corresponding filenames are saved alongside the index.

During search:

* The user's text query is converted into a CLIP text embedding.
* FAISS compares it with all stored image embeddings.
* The closest matching photos are returned.

```text
Photos
   ↓
CLIP Image Encoder
   ↓
Image Embeddings
   ↓
FAISS Index
   ↓
Saved to Disk

Text Query
   ↓
CLIP Text Encoder
   ↓
Text Embedding
   ↓
FAISS Search
   ↓
Matching Photos
```

---

## Technologies Used

* Python
* CLIP (OpenAI)
* FAISS
* Gradio
* PyTorch
* Pillow

---

## Project Structure

```text
PROJECT_imagedescriptor/

├── APP.py              # Gradio application
├── Buildindex.py       # Creates image embeddings and FAISS index
├── Requirements.txt    # Python dependencies
├── README.md
├── photos/             # Your photos
└── index/              # Generated search index
    ├── photos.index
    ├── filenames.json
    └── config.json
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd PROJECT_imagedescriptor
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

> **Note:** This project uses `Requirements.txt` with a capital **R**.

```bash
pip install -r Requirements.txt
```

---

## 4. Add Your Photos

Create a folder named `photos` inside the project.

```text
PROJECT_imagedescriptor/
└── photos/
    ├── image1.jpg
    ├── image2.png
    └── holiday/
        └── beach.jpg
```

Supported formats:

* JPG
* JPEG
* PNG
* BMP
* WEBP

Subfolders are supported.

---

## 5. Build the Image Index

Generate CLIP embeddings for your photos.

```bash
python Buildindex.py --photos_dir "./photos" --output_dir "./index"
```

If your folder name contains spaces, wrap the path in quotes.

Example:

```bash
python Buildindex.py --photos_dir "./sample photos" --output_dir "./index"
```

After this finishes, an `index` folder will be created containing:

* `photos.index`
* `filenames.json`
* `config.json`

Run this step again whenever you add or remove photos.

---

## 6. Launch the Application

Start the Gradio app.

```bash
python APP.py --index_dir "./index"
```

The terminal will display a local URL similar to:

```text
http://127.0.0.1:7860
```

Open that link in your browser.

---

## Example Searches

Try queries like:

```text
a person standing near a car
```

```text
a sunset over the mountains
```

```text
a black dog running
```

The app returns the most similar images along with similarity scores.

---

## Similarity Search

The project uses **FAISS IndexFlatIP** for exact inner-product similarity search.

This works well for small and medium-sized image collections while providing fast and accurate retrieval.

---

## Limitations

CLIP performs well for semantic descriptions but may struggle with:

* Exact object counts (`exactly three dogs`)
* Fine spatial relationships (`the dog on the left side of the car`)
* Very detailed scene constraints

The selected CLIP model (`ViT-B/32` by default) balances speed and accuracy.

---

## Possible Improvements

* Image-to-image search
* Filters (date, folder, file type)
* Duplicate photo detection
* Camera metadata search
* Larger CLIP models
* Approximate FAISS indexes for huge collections

---

## Deployment

The application can be deployed on platforms that support **Gradio**, such as **Hugging Face Spaces**.

A typical deployment includes:

1. Creating a Gradio Space.
2. Uploading `APP.py`.
3. Uploading `Requirements.txt`.
4. Adding a small sample photo collection.
5. Building the FAISS index.
6. Launching the app.

---

## What I Learned

This project helped me work with:

* CLIP embeddings
* Vector similarity search
* FAISS indexing
* PyTorch
* Gradio
* Building an end-to-end AI application using pretrained models

---
## SHIVI SANJAY

## Author

**Shivi Sanjay**
