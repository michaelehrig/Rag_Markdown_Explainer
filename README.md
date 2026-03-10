# 📚 Beginner-Friendly RAG Markdown Explainer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Google-Gemini-orange)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-purple)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

A compact **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Google Gemini**, and **FAISS**.

This project loads documents from a folder, builds a vector index, retrieves relevant context for a topic, and generates a **beginner-friendly explanation saved as Markdown**.

It demonstrates the core building blocks of modern LLM applications:

- document ingestion
- embeddings and vector search
- retrieval-augmented prompting
- structured LLM output
- Markdown generation

The goal is to keep the system **small, readable, and educational** while still doing something genuinely useful.

---

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Example Output](#-example-output)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Command-Line Options](#-command-line-options)
- [Supported File Types](#-supported-file-types)
- [Main Components](#-main-components)
- [Ideas for Improvement](#-ideas-for-improvement)
- [License](#-license)

---

## ✨ Features

✔ Simple **local-folder RAG pipeline**  
✔ Works with **PDF, Markdown, and text documents**  
✔ Uses **FAISS vector search**  
✔ Uses **Gemini embeddings + LLM generation**  
✔ Produces **clean Markdown output**  
✔ Enforces **structured LLM responses with Pydantic**  
✔ Reusable vector index with optional rebuild flag

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Rag_Markdown_Explainer.git
cd Rag_Markdown_Explainer
````

---

### 2️⃣ Install dependencies

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters \
langchain-google-genai faiss-cpu pydantic pypdf
```

---

### 3️⃣ Set your Gemini API key

#### macOS / Linux

```bash
export GEMINI_API_KEY="your_api_key_here"
```

#### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

---

### 4️⃣ Add source documents

Create a folder containing the documents you want to use as knowledge sources.

Example:

```
source_docs/
├── article.txt
├── notes.md
└── handbook.pdf
```

---

### 5️⃣ Run the script

```bash
python explainer.py
```

You will be prompted for:

```
Enter a topic to explain:
Enter the output markdown filename:
Enter source folder name for RAG:
Enter folder for FAISS index:
```

Example:

```
Enter a topic to explain: vector databases
Enter the output markdown filename: vector_db.md
Enter source folder name for RAG: source_docs
Enter folder for FAISS index: faiss_index
```

The explanation will be saved as a Markdown file.

---

## 📄 Example Output

Generated Markdown output:

```markdown
# Explanation of Vector Databases

## Simple Explanation

A vector database stores numerical representations of data so that similar items can be found quickly.

## Analogy

Imagine a library where books are not organized by title, but by meaning...

## Key Points

- Vectors represent semantic meaning
- Similar vectors are located near each other
- Vector search enables semantic retrieval
- They are widely used in RAG systems
- They improve search relevance

## Explanation

Vector databases store embeddings—numerical representations of text, images, or other data...

## Sources Used

- source_docs/notes.md
- source_docs/handbook.pdf (page 3)
```

---

## 📂 Project Structure

Example repository layout:

```
rag-markdown-explainer/
│
├── explainer.py
├── README.md
│
├── source_docs/
│   ├── notes.md
│   ├── research.txt
│   ├── handbook.pdf
│   └── faiss_index/
│
└── output.md
```

---

## ⚙️ How It Works

### 1️⃣ Document Loading

The script recursively scans a folder and loads:

- `.txt`
- `.md`
- `.pdf`

PDF files are processed page-by-page using **PyPDFLoader**.

Each document receives metadata:

- source path
- file name
- file type

---

### 2️⃣ Chunking

Documents are split using:

```
RecursiveCharacterTextSplitter
```

Configuration:

```
chunk_size = 1000
chunk_overlap = 200
```

Chunking improves retrieval accuracy and embedding quality.

---

### 3️⃣ Embeddings & Vector Store

The project uses:

```
GoogleGenerativeAIEmbeddings
models/gemini-embedding-001
```

Chunks are embedded and stored in a **FAISS vector index**.

If the index already exists, it will be loaded automatically.

---

### 4️⃣ Retrieval

When a topic is entered:

1. The query is embedded
2. FAISS performs similarity search
3. The **top 4 chunks** are retrieved
4. Context is assembled for the LLM

---

### 5️⃣ LLM Generation

The system prompts **Gemini 2.5 Flash** with:

- the topic
- retrieved context
- instructions for structured output

The model returns a response matching the Pydantic schema:

```
Explanation
 ├── simple_explanation
 ├── analogy
 ├── key_points
 └── extensive_explanation
```

---

### 6️⃣ Markdown Formatting

The structured output is converted into a clean Markdown document containing:

- title
- simple explanation
- analogy
- key points
- long explanation
- list of sources

---

## 🧰 Command-Line Options

### `--rebuild-index`

Rebuild the FAISS index instead of reusing the existing one.

Useful when:

- documents were added
- documents were modified
- embeddings need to be refreshed

Example:

```bash
python explainer.py --rebuild-index
```

---

## 📄 Supported File Types

Currently supported:

| File Type | Supported |
| --------- | --------- |
| `.txt`    | ✅         |
| `.md`     | ✅         |
| `.pdf`    | ✅         |

Additional loaders could easily be added for:

* `.docx`
* `.html`
* `.csv`
* `.json`

---

## 🧩 Main Components

### `Explanation`

Pydantic schema that enforces the structure of the LLM output.

---

### `load_documents_from_folder()`

Recursively loads supported documents.

---

### `split_documents()`

Splits documents into chunks for embedding.

---

### `build_or_load_vectorstore()`

Creates or loads the FAISS vector index.

---

### `add_context()`

Retrieves relevant document chunks for the topic.

---

### `to_markdown()`

Formats the structured explanation into Markdown.

---

### `main()`

Coordinates the pipeline:

```
documents → chunks → embeddings → retrieval → generation → markdown
```

---

## 🔧 Ideas for Improvement

Potential enhancements:

- Add **DOCX / HTML loaders**
- Convert inputs from `input()` to **full CLI arguments**
- Add **configurable chunk sizes**
- Support **multiple retrieval strategies**
- Include **retrieval scores**
- Add **logging**
- Build a **web UI**
- Integrate **evaluation metrics**

---

## 📜 License

MIT License

Feel free to use, modify, and extend this project.

---

## 🙌 Acknowledgment

This repository demonstrates a **minimal but realistic RAG workflow**.

Instead of hiding everything behind large frameworks, the goal is to show how the core components fit together:

```
documents → embeddings → retrieval → generation
```

Sometimes the best way to understand modern AI systems is not another abstraction layer — but a small script that you can read end-to-end in a few minutes.
