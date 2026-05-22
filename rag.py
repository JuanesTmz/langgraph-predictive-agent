"""
RAG (Retrieval-Augmented Generation) — infraestructura de recuperación.

Responsabilidades:
  1. Leer los documentos de cada departamento.
  2. Dividirlos en fragmentos (chunks) manejables.
  3. Convertirlos en vectores usando embeddings de Ollama.
  4. Almacenarlos en un vector store en memoria por departamento.
  5. Exponer la función `retrieve` para buscar fragmentos relevantes.
"""

from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Configuración ─────────────────────────────────────────────────────────────

DOCS_PATH = Path(__file__).parent / "docs"
EMBEDDING_MODEL = "qwen2.5:3b"
OLLAMA_HOST = "http://localhost:11434"


class _OllamaEmbeddings(Embeddings):
    """
    Wrapper ligero sobre la API REST de Ollama para generar embeddings.

    Por qué no usamos OllamaEmbeddings de langchain-ollama:
      En langchain-ollama>=1.1.0 la clase crea un httpx.AsyncClient durante
      __init__, lo que en Python 3.13 + Windows bloquea al cargar el contexto
      SSL de certifi. Al llamar directamente con requests (http local, sin SSL)
      evitamos esa inicialización problemática.
    """

    def __init__(self, model: str, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self._url = f"{host}/api/embed"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            self._url,
            json={"model": self.model, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

# Cada departamento tiene su propio archivo de conocimiento
DEPARTMENT_DOCS: dict[str, str] = {
    "IT":             "it_manual.txt",
    "RRHH":           "rrhh_politicas.txt",
    "Vacaciones":     "vacaciones_politica.txt",
    "Incapacidades":  "incapacidades_procedimiento.txt",
}

# ── Construcción de vectorstores ──────────────────────────────────────────────

def build_vectorstores() -> dict[str, InMemoryVectorStore]:
    """
    Lee todos los documentos, los fragmenta y construye un InMemoryVectorStore
    independiente por cada departamento.

    Por qué un store por departamento:
      - El Router Agent ya filtró el tema → buscar solo en el corpus relevante
        reduce el ruido y mejora la precisión de recuperación.
    """
    embeddings = _OllamaEmbeddings(model=EMBEDDING_MODEL)

    # chunk_size=500 chars / overlap=50 para preservar contexto entre fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    stores: dict[str, InMemoryVectorStore] = {}

    for dept, filename in DEPARTMENT_DOCS.items():
        raw_text = (DOCS_PATH / filename).read_text(encoding="utf-8")

        # Cada chunk se convierte en un Document con metadata de origen
        chunks: list[Document] = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"department": dept, "source": filename}],
        )

        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(chunks)

        stores[dept] = store
        print(f"  [OK] [{dept}] {len(chunks)} fragmentos indexados")

    return stores


# ── Función de recuperación ───────────────────────────────────────────────────

def retrieve(query: str, department: str, vectorstores: dict, k: int = 3) -> str:
    """
    Busca los k fragmentos más similares semánticamente a `query`
    dentro del vectorstore del departamento indicado.

    Devuelve el contexto como string numerado [1], [2], [3]
    para que el Synthesizer Agent pueda citarlo fácilmente.
    """
    results: list[Document] = vectorstores[department].similarity_search(query, k=k)

    if not results:
        return "No se encontró información relevante en la base de conocimiento."

    return "\n\n".join(
        f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(results)
    )
