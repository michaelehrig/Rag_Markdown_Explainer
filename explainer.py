from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Structured output schema
# --------------------------------------------------

class Explanation(BaseModel):
    simple_explanation: str = Field(description="A plain-English explanation")
    analogy: str = Field(description="A helpful analogy for beginners")
    key_points: list[str] = Field(description="Five short key points")
    extensive_explanation: str = Field(
        description="A longer plain-English explanation with more details"
    )

# --------------------------------------------------
# Document loading
# --------------------------------------------------

def load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        page_content=text,
        metadata={
            "source": str(path),
            "file_name": path.name,
            "file_type": path.suffix.lower(),
        },
    )

def load_pdf_file(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    # Make sure metadata contains consistent source info
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["file_name"] = path.name
        doc.metadata["file_type"] = path.suffix.lower()
    return docs

def load_documents_from_folder(folder: str) -> list[Document]:
    base = Path(folder)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Source folder does not exist or is not a directory: {folder}")

    documents: list[Document] = []

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            documents.append(load_text_file(path))
        elif suffix == ".pdf":
            documents.extend(load_pdf_file(path))

    if not documents:
        raise ValueError(
            "No supported files found. Add .txt, .md, or .pdf files to the source folder."
        )

    return documents

# --------------------------------------------------
# Vector store creation / loading / data chunking
# --------------------------------------------------

def build_or_load_vectorstore(docs: list[Document], index_path: str, rebuild: bool = False) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    index_dir_path = Path(index_path)

    if index_dir_path.exists() and not rebuild:
        print(f"Loading existing FAISS index from: {index_dir_path}")
        return FAISS.load_local(
            str(index_dir_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Building FAISS index from documents...")
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} source documents")
    print(f"Created {len(chunks)} chunks")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_dir_path))
    print(f"Saved FAISS index to: {index_dir_path}")
    return vectorstore

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)

# --------------------------------------------------
# Retrieval helpers
# --------------------------------------------------

def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def add_context(data: dict) -> dict:
    query_topic = data["topic"]
    retrieved_docs = data["retriever"].invoke(query_topic)

    context_blocks = []
    source_lines = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")

        if page is not None:
            source_label = f"{source} (page {page + 1})"
        else:
            source_label = source

        context_blocks.append(
            f"[Source {i}: {source_label}]\n{doc.page_content}"
        )
        source_lines.append(source_label)

    return {
        "topic": query_topic,
        "context": "\n\n".join(context_blocks),
        "sources": unique_preserve_order(source_lines),
    }

# --------------------------------------------------
# Markdown formatting
# --------------------------------------------------

def to_markdown(data: dict) -> str:
    topic_value = data["topic"]
    result: Explanation = data["explanation"]
    sources: list[str] = data["sources"]

    lines = [
        f"# Explanation of {topic_value}",
        "",
        "## Simple Explanation",
        "",
        result.simple_explanation,
        "",
        "## Analogy",
        "",
        result.analogy,
        "",
        "## Key Points",
        "",
    ]

    lines.extend(f"- {point}" for point in result.key_points)

    lines.extend([
        "",
        "## Explanation",
        "",
        result.extensive_explanation,
        "",
        "## Sources Used",
        "",
    ])

    if sources:
        lines.extend(f"- {source}" for source in sources)
    else:
        lines.append("- No retrieved sources were used.")

    lines.append("")
    return "\n".join(lines)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    # Argument Parsing

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    # User Input

    topic = input("Enter a topic to explain: ").strip()
    filename = input("Enter the output markdown filename (e.g. explanation.md): ").strip()
    source_dir = input("Enter source folder name for RAG: ").strip()
    index_dir_base = input(
        "Enter folder for FAISS index (relative to source folder): "
    ).strip() or "faiss_index"
    index_dir = source_dir+"/"+index_dir_base

    # Environmental Variables

    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please export your Gemini API key first."
        )
    
    # Prompt

    prompt = ChatPromptTemplate.from_template(
"""
You are explaining a concept to a beginner.

Use the retrieved context below to explain the topic.
Prefer the retrieved material when it is relevant.
If the retrieved context is incomplete, still answer helpfully, but do not invent fake source claims.

Topic:
{topic}

Retrieved context:
{context}

Return:
- a simple explanation
- a helpful analogy
- five short key points
- a more in depth explanation, usually at least 5 times as long as the simple explanation
"""
    )

    # Model

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    structured_model = model.with_structured_output(Explanation)

    # Build retrieval pipeline

    all_docs = load_documents_from_folder(source_dir)
    vectorstore = build_or_load_vectorstore(all_docs, index_dir, rebuild=args.rebuild_index)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Build runnables and final chain

    retrieve_context = RunnableLambda(add_context)

    generate_explanation = RunnableParallel(
        topic=RunnableLambda(lambda x: x["topic"]),
        context=RunnableLambda(lambda x: x["context"]),
        sources=RunnableLambda(lambda x: x["sources"]),
        explanation=prompt | structured_model,
    )

    markdown_formatter = RunnableLambda(to_markdown)

    chain = retrieve_context | generate_explanation | markdown_formatter

    # Run and save

    markdown_output = chain.invoke({"topic": topic, "retriever": retriever})

    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print(f"\nMarkdown file saved as: {filename}")


if __name__ == "__main__":
    main()

        
        