"""
Infraestructura RAG: embeddings, vectorstores y recuperación de documentos.

Responsabilidades:
  1. Leer los documentos de cada departamento desde docs/.
  2. Dividirlos en fragmentos (chunks) de ~500 caracteres.
  3. Convertirlos en vectores usando sentence-transformers (multilingüe).
  4. Almacenarlos en un InMemoryVectorStore por departamento.
  5. Exponer retrieve() para buscar los fragmentos más relevantes a una pregunta.

Por qué sentence-transformers en vez del embedding de Ollama:
  - Modelo entrenado específicamente para similitud semántica
  - Soporte nativo de español y 50+ idiomas en un solo modelo
  - Sin dependencia del servidor Ollama para la capa de embeddings
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path(__file__).parent.parent / "docs"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

DEPARTMENT_DOCS: dict[str, str] = {
    "IT":            "it_manual.txt",
    "RRHH":          "rrhh_politicas.txt",
    "Vacaciones":    "vacaciones_politica.txt",
    "Incapacidades": "incapacidades_procedimiento.txt",
}


class _STEmbeddings(Embeddings):
    """Adapter de sentence-transformers compatible con la interfaz Embeddings de LangChain."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def build_vectorstores() -> dict[str, InMemoryVectorStore]:
    """Construye un vectorstore por departamento con los documentos indexados."""
    embeddings = _STEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    stores: dict[str, InMemoryVectorStore] = {}

    for dept, filename in DEPARTMENT_DOCS.items():
        raw_text = (DOCS_PATH / filename).read_text(encoding="utf-8")
        chunks: list[Document] = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"department": dept, "source": filename}],
        )
        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(chunks)
        stores[dept] = store
        print(f"  [OK] [{dept}] {len(chunks)} fragmentos indexados")

    return stores


def retrieve(query: str, department: str, vectorstores: dict, k: int = 3) -> str:
    """Devuelve los k fragmentos más similares a query dentro del departamento dado."""
    results: list[Document] = vectorstores[department].similarity_search(query, k=k)

    if not results:
        return "No se encontró información relevante en la base de conocimiento."

    return "\n\n".join(
        f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(results)
    )


def rag_agent(state: dict, vectorstores: dict) -> dict:
    """
    NODO — RAG Agent.

    Busca en el vectorstore del departamento los 3 fragmentos más relevantes
    para la pregunta del empleado y los escribe en retrieved_context.
    """
    context = retrieve(
        query=state["query"],
        department=state["department"],
        vectorstores=vectorstores,
        k=3,
    )
    return {"retrieved_context": context}
