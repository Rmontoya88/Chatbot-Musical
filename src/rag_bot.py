
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CARGA INICIAL (se hace una sola vez)
# =========================
print("📂 Cargando índice y chunks...")
indice = faiss.read_index("indice_parrafos.faiss")

with open("chunks_parrafos.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Índice cargado: {indice.ntotal} vectores")

print("Cargando modelo de embeddings...")
modelo_embeddings = SentenceTransformer('intfloat/e5-base-v2')
print("✅ Modelo listo")


# =========================
# BUSQUEDA
# =========================
def buscar_chunks_relevantes(pregunta, top_k=5):
    embedding = modelo_embeddings.encode(
        [pregunta], convert_to_numpy=True
    ).astype('float32')

    faiss.normalize_L2(embedding)
    distancias, indices = indice.search(embedding, top_k)

    resultados = []
    for dist, idx in zip(distancias[0], indices[0]):
        chunk = chunks[idx]

        if isinstance(chunk, tuple):
            titulo, artista, texto = chunk
        else:
            titulo, artista, texto = "desconocido", "desconocido", chunk

        resultados.append({
            "titulo": titulo,
            "artista": artista,
            "texto": texto,
            "score": float(1 - dist)
        })

    return resultados


# =========================
# CLASE BOT
# =========================
class RAGBot:
    def __init__(self):
        self.historial = []

    def responder(self, mensaje):
        resultados = buscar_chunks_relevantes(mensaje)

        contexto = "\n\n".join([
            f"""
CANCION: {r['titulo']}
ARTISTA: {r['artista']}
LETRA: {r['texto']}
"""
            for r in resultados
        ])

        prompt = f"""
Eres un recomendador musical.

Usa los documentos para recomendar EXACTAMENTE 3 canciones.
No inventes información.

Formato:

1. Canción:
   Artista:
   Explicación:

2. Canción:
   Artista:
   Explicación:

3. Canción:
   Artista:
   Explicación:

Documentos:
{contexto}

Pregunta: {mensaje}
"""

        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )

        return response.choices[0].message.content