"""
Agentes NLP: preprocesamiento con spaCy y análisis de sentimiento con Transformers.

Estos dos agentes corren PRIMERO en el pipeline, antes del LLM, para enriquecer
el estado con información que los agentes siguientes pueden aprovechar.

  preprocess_agent  →  extrae entidades y palabras clave (spaCy)
  sentiment_agent   →  detecta si el empleado está frustrado (HuggingFace Transformers)

Por qué usar Transformers en vez del LLM para el sentimiento:
  - Inferencia local sin necesitar el servidor Ollama
  - Clasificación determinista y reproducible
  - Más rápido que invocar un LLM completo para una sola etiqueta
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import spacy
from transformers import pipeline

# ── spaCy ────────────────────────────────────────────────────────────────────

try:
    _NLP = spacy.load("es_core_news_sm")
except OSError:
    from spacy.cli import download as _dl
    print("[spaCy] Descargando modelo es_core_news_sm ...")
    _dl("es_core_news_sm")
    _NLP = spacy.load("es_core_news_sm")

# ── HuggingFace Transformers ──────────────────────────────────────────────────

print("[Transformers] Cargando modelo de sentimiento ...")
_SENTIMENT = pipeline(
    "text-classification",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    top_k=1,
)


# ── Agentes ───────────────────────────────────────────────────────────────────

def preprocess_agent(state: dict) -> dict:
    """
    NODO — Preprocesador NLP (spaCy).

    Lee la pregunta del empleado y extrae:
      - Entidades nombradas: personas, organizaciones, siglas (ej: "VPN", "IT")
      - Palabras clave: sustantivos y verbos relevantes, sin stopwords

    Esta información llega al Synthesizer para personalizar la respuesta.
    """
    doc = _NLP(state["query"])

    entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
    keywords = sorted({
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop and len(token.text) > 2
    })

    return {"entities": entities, "keywords": keywords}


def sentiment_agent(state: dict) -> dict:
    """
    NODO — Análisis de sentimiento (HuggingFace Transformers).

    Clasifica el tono emocional de la consulta y define un nivel de urgencia:
      negative + score > 0.75  →  urgencia high   (respuesta muy empática)
      negative + score ≤ 0.75  →  urgencia medium  (respuesta empática)
      positive / neutral       →  urgencia low     (respuesta estándar)

    El Synthesizer Agent ajusta su tono en base a esta urgencia.
    """
    output = _SENTIMENT(state["query"])[0][0]
    label: str   = output["label"].lower()
    score: float = output["score"]

    if label == "negative" and score > 0.75:
        urgency = "high"
    elif label == "negative":
        urgency = "medium"
    else:
        urgency = "low"

    return {"sentiment": label, "urgency_level": urgency}
