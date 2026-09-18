# Photo Search — Search Photos Using Text

This project lets you search through a collection of photos by simply describing what you are looking for.

For example, instead of searching for a filename like `IMG_2048.jpg`, you can type:

> `a dog running on the beach`

The application uses **CLIP** to understand both the images and the text query. **FAISS** is then used to find the images that are most similar to the query. The results are displayed through a simple **Gradio** web interface.

## How the Project Works

The project has two main parts:

1. **Building the image index**
2. **Searching the images**

When the index is created, every image in the selected folder is passed through the CLIP image encoder. This produces an embedding (a numerical representation of the image).

These embeddings are stored in a FAISS index along with the corresponding image filenames.

When a user searches for something, the text is also converted into an embedding using CLIP. FAISS compares this embedding with the stored image embeddings and returns the closest matches.

```text
Photos
   ↓
CLIP image encoder
   ↓
Image embeddings
   ↓
FAISS index
   ↓
Stored on disk


Text entered by user
   ↓
CLIP text encoder
   ↓
Text embedding
   ↓
FAISS similarity search
   ↓
Matching photos
```

One useful part of this approach is that the project doesn't need a separate dataset with manually created labels. The CLIP model has already learned relationships between images and text.

## Technologies Used

* **Python**
* **CLIP** – for converting images and text into embeddings
* **FAISS** – for similarity search
* **Gradio** – for the web interface
* **Pillow** – for loading and processing images
* **PyTorch** – used by the CLIP model

## Setup

Clone the repository and move into the project folder:

```bash
git clone <your-repo-url>
cd clip-image-search
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Add your photos

Create a `photos` folder and put the images you want to search inside it.

The program also checks subfolders, so you can organize your photos however you want.

Supported formats include:

* JPG
* JPEG
* PNG
* BMP
* WEBP

### 2. Build the image index

Run:

```bash
python build_index.py --photos_dir ./photos --output_dir ./index
```

This goes through the photos, generates their CLIP embeddings, and saves the FAISS index.

You need to run this again if you add or remove photos from your collection.

The generated files are stored inside the `index` folder.

### 3. Start the application

Run:

```bash
python app.py --index_dir ./index
```

The terminal will show a local Gradio URL. Open that URL in your browser.

Enter a description such as:

```text
a person standing near a car
```

or:

```text
a sunset over the mountains
```

The application will show the photos that are closest to the query.

## Project Structure

```text
clip-image-search/
│
├── build_index.py       # Creates the image embeddings and FAISS index
├── app.py               # Gradio application
├── requirements.txt     # Python dependencies
├── README.md
│
├── photos/              # Photos used for searching
│
└── index/               # Generated search index
    ├── photos.index
    ├── filenames.json
    └── config.json
```

## Similarity Search

The project uses FAISS to compare the text embedding with the image embeddings.

`IndexFlatIP` is used for the search. It performs an exact inner-product search, which works well for a relatively small photo collection.

The application also displays a similarity score with the search results so that the user can get an idea of how closely each image matches the query.

## Limitations

There are some things that this project does not handle perfectly.

For example, CLIP generally works well with queries such as:

```text
a dog on a beach
```

but can have difficulty with more specific requests such as:

```text
exactly three dogs
```

It can also struggle with detailed spatial relationships, for example:

```text
the dog on the left side of the car
```

The choice of CLIP model also affects the results. A smaller model such as `ViT-B/32` is useful when speed and lower resource usage are important, while larger models can provide better results but require more computation.

The current FAISS setup performs an exact search. This is reasonable for a smaller collection of images, but for very large collections, approximate search methods such as `IndexIVFFlat` or `IndexHNSWFlat` could be considered.

## Possible Improvements

Some features that could be added later are:

### Image-to-Image Search

Instead of typing a description, the user could upload an image and find photos that look similar to it.

### Filters

Search results could be combined with filters such as:

* Date
* File type
* Folder
* Image location
* Camera information

### Duplicate Detection

Image embeddings could also be used to find photos that are very similar to each other and identify possible duplicates.

### Better Search Models

Different CLIP checkpoints could be tested to see how they affect search accuracy and speed.

## Deployment

The application can be deployed using platforms that support Gradio applications.

For example, a sample version can be hosted on **Hugging Face Spaces**.

A basic deployment would require:

1. Creating a new Space with the Gradio SDK.
2. Adding `app.py`.
3. Adding `requirements.txt`.
4. Adding a small sample image collection.
5. Creating the FAISS index for those images.
6. Testing the application through the generated public URL.

For a portfolio project, it is better to use a small set of sample images rather than uploading a complete personal photo library.

## What I Learned

Through this project, I worked with:

* Image and text embeddings
* CLIP
* Vector similarity search
* FAISS
* Python
* Gradio
* Building a simple search application
* Working with pretrained models without training a model from scratch

The main idea behind the project was to make photo searching more convenient by allowing users to describe what they want instead of remembering filenames or manually going through every image.

## Author
# SHIVI SANJAY
