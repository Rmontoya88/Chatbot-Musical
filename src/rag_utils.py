# Funciones para RAG
import random
import re
import numpy as np
from pymongo import MongoClient
import random
import re
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from dotenv import load_dotenv
load_dotenv()


# ==============================
# MONGO
# ==============================



from src.mongoDB import MongoReader

import os
from dotenv import load_dotenv

load_dotenv()

mongo_reader = MongoReader(
    uri=os.getenv("MONGO_URI"),
    db_name=os.getenv("DB_NAME"),
    collection_name=os.getenv("COLLECTION_NAME"),
)

# ==============================
# CHUNKING
# ==============================

class MusicChunker:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = self.db[collection_name]

        self.chunks_fijo = []
        self.chunks_oraciones = []
        self.chunks_parrafos = []


    def chunking_fijo(self, texto, tamano_chunk=300, overlap=50):
        chunks = []
        inicio = 0

        while inicio < len(texto):
            fin = inicio + tamano_chunk
            chunk = texto[inicio:fin].strip()
            if chunk:
                chunks.append(chunk)
            inicio = fin - overlap

        return chunks

    def chunking_oraciones(self, texto, oraciones_por_chunk=3, overlap_oraciones=1):
        oraciones = re.split(r'(?<=[.!?])\s+', texto.strip())
        oraciones = [o.strip() for o in oraciones if o.strip()]

        chunks = []
        i = 0

        while i < len(oraciones):
            grupo = oraciones[i:i + oraciones_por_chunk]
            chunk = " ".join(grupo)
            if chunk:
                chunks.append(chunk)
            i += oraciones_por_chunk - overlap_oraciones

        return chunks

    def chunking_parrafos(self, texto, min_longitud=50):
        parrafos = re.split(r'\n\s*\n', texto)
        parrafos = [p.strip() for p in parrafos if p.strip()]

        chunks = []
        buffer = ""

        for p in parrafos:
            if len(buffer) + len(p) < min_longitud * 3:
                buffer += " " + p
            else:
                if buffer.strip():
                    chunks.append(buffer.strip())
                buffer = p

        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks

    # ==============================
    # MONGO
    # ==============================
    def get_documents(self, limit=None):
        query = self.col.find(
            {},
            {"_id": 0, "titulo": 1, "artista": 1, "letra": 1}
        )

        if limit:
            query = query.limit(limit)

        return list(query)

    # ==============================
    # PIPELINE
    # ==============================
    def process(self, limit=1000):
        docs = self.get_documents(limit)

        for doc in docs:
            texto = doc.get("letra", "")
            titulo = doc.get("titulo", "")
            artista = doc.get("artista", "")

            if not texto:
                continue

            cf = self.chunking_fijo(texto, 400, 80)
            co = self.chunking_oraciones(texto, 3, 1)
            cp = self.chunking_parrafos(texto)

            for c in cf:
                self.chunks_fijo.append((titulo, artista, c))

            for c in co:
                self.chunks_oraciones.append((titulo, artista, c))

            for c in cp:
                self.chunks_parrafos.append((titulo, artista, c))

    # ==============================
    # ANALISIS
    # ==============================
    def analizar(self, nombre, lista):
        tamanos = [len(x[2]) for x in lista]

        idx = random.randint(0, len(lista) - 1)

        print(f"\n{nombre}:")
        print(f"   Total chunks: {len(lista)}")
        print(f"   Tamano promedio: {np.mean(tamanos):.0f} caracteres")
        print(f"   Min/Max: {min(tamanos)}/{max(tamanos)}")
        print(f"   Ejemplo random idx={idx}")
        print(f"   [{lista[idx][0]} - {lista[idx][1]}]")
        print(f"   {lista[idx][2][:120]}...")

    def comparar(self):
        print("\nCOMPARACION DE ESTRATEGIAS DE CHUNKING")
        print("=" * 60)

        self.analizar("Fijo (400 chars)", self.chunks_fijo)
        self.analizar("Por oraciones", self.chunks_oraciones)
        self.analizar("Por parrafos", self.chunks_parrafos)


# ==============================
# Embedings y Faiss
# ==============================

class RAGPipeline:
    def __init__(self, indice_path, chunks_path, embeddings_path, model_name="intfloat/e5-base-v2"):
        """Inicializa el pipeline RAG con búsqueda híbrida"""
        from dotenv import load_dotenv
        import os

        # Cargar variables de entorno con ruta absoluta
        env_path = r"C:\Users\rmont\Downloads\proyecto3_chatbot_musical\proyecto3_chatbot_musical\src\.env"
        load_dotenv(env_path)

        # Verificar que se cargó
        api_key = os.getenv('OPENAI_API_KEY')
        print(f"🔑 API Key detectada: {'Sí' if api_key else 'NO'}")

        print("📂 Cargando índice y chunks...")
        self.indice = faiss.read_index(indice_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embeddings_originales = np.load(embeddings_path)

        print(f"✅ Índice cargado: {self.indice.ntotal} vectores")
        print("Cargando modelo de embeddings...")
        self.modelo_embeddings = SentenceTransformer(model_name)
        print("✅ Modelo listo")

    def buscar_chunks_relevantes(self, pregunta, top_k=3):
        """Búsqueda semántica simple con FAISS"""
        embedding_pregunta = self.modelo_embeddings.encode(
            [pregunta],
            convert_to_numpy=True
        ).astype('float32')
        faiss.normalize_L2(embedding_pregunta)

        distancias, indices = self.indice.search(embedding_pregunta, top_k)

        resultados = []
        for dist, idx in zip(distancias[0], indices[0]):
            cancion = self.chunks[idx][0]
            artista = self.chunks[idx][1]
            texto = self.chunks[idx][2]
            resultados.append({
                "cancion": cancion,
                "artista": artista,
                "letra": texto,
                "score": float(1 - dist),
                "indice": int(idx)
            })
        return resultados

    def buscar_chunks_hibrido(self, pregunta, top_k=3):
        """Búsqueda híbrida: metadatos + embeddings"""
        pregunta_lower = pregunta.lower()

        # Detectar chunks cuyo artista o título aparece en la pregunta
        candidatos_indices = []
        for i, chunk in enumerate(self.chunks):
            cancion = chunk[0].lower()
            artista = chunk[1].lower()
            palabras_artista = artista.split()
            palabras_cancion = [p for p in cancion.split() if len(p) > 3]

            if any(p in pregunta_lower for p in palabras_artista) or \
                    any(p in pregunta_lower for p in palabras_cancion):
                candidatos_indices.append(i)

        if candidatos_indices:
            print(f"   🎯 Filtro por metadatos: {len(candidatos_indices)} chunks del artista/canción")

            # Tomar embeddings del subconjunto
            sub_embeddings = self.embeddings_originales[candidatos_indices].copy().astype('float32')
            faiss.normalize_L2(sub_embeddings)

            embedding_pregunta = self.modelo_embeddings.encode(
                ["query: " + pregunta], convert_to_numpy=True
            ).astype('float32')
            faiss.normalize_L2(embedding_pregunta)

            scores = sub_embeddings @ embedding_pregunta.T
            scores = scores.flatten()
            top_local = np.argsort(scores)[::-1][:top_k]

            resultados = []
            for rank_i in top_local:
                real_i = candidatos_indices[rank_i]
                chunk = self.chunks[real_i]
                resultados.append({
                    "cancion": chunk[0],
                    "artista": chunk[1],
                    "letra": chunk[2],
                    "score": float(scores[rank_i]),
                    "indice": real_i
                })
            return resultados

        # Fallback: FAISS global
        print("   ⚠️ Sin match por metadatos, usando FAISS global")
        return self.buscar_chunks_relevantes("query: " + pregunta, top_k)

    def generar_con_openai(self, contexto, pregunta):
        """Genera respuesta usando OpenAI"""
        from openai import OpenAI
        import os

        # Obtener API key del entorno
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Error: OPENAI_API_KEY no encontrada en variables de entorno"

        client = OpenAI(api_key=api_key)

        prompt = f"""
    Eres un analista de letras de canciones.
    Tu tarea es evaluar TODAS las canciones del contexto.

    OBJETIVO:
    Identificar qué canciones cumplen con la condición de la pregunta.

    REGLAS:
    - Considera tanto coincidencia explícita como implícita.
    - Usa razonamiento semántico, no solo palabras exactas.

    FORMATO:
    Canciones que cumplen:
    1. <Canción> - <Artista>
       Explicación breve

    Evidencia:
    - "fragmento relevante"

    CONTEXTO:
    {contexto}

    PREGUNTA:
    {pregunta}

    RESPUESTA:
    """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message.content

    def rag_completo(self, pregunta, top_k=3, modelo="openai"):
        """Pipeline RAG completo"""
        print(f"\n{'=' * 60}")
        print(f"PREGUNTA: {pregunta}")
        print(f"Modelo: {modelo} | Top-K: {top_k}")
        print(f"{'=' * 60}")

        # Búsqueda híbrida
        resultados = self.buscar_chunks_hibrido(pregunta, top_k)

        print(f"\nChunks recuperados:")
        for i, r in enumerate(resultados):
            print(f"   [{i + 1}] {r['cancion']} - {r['artista']} | Score: {r['score']:.4f}")

        # Crear contexto
        contexto = "\n\n".join([
            f"[Chunk {i + 1}]\nCanción: {r['cancion']}\nArtista: {r['artista']}\nLetra: {r['letra']}"
            for i, r in enumerate(resultados)
        ])

        print(f"\nGenerando respuesta...")

        # Generar respuesta
        if modelo == "openai":
            respuesta = self.generar_con_openai(contexto, pregunta)
        else:
            respuesta = "Modelo no reconocido"

        print(f"\nRESPUESTA: {respuesta}")
        print(f"{'=' * 60}")

        return respuesta

# ==============================
# Modelo DistilBERT QA
# ==============================


class QAPipeline:
    def __init__(self, indice_path, chunks_path, embeddings_path,
                 qa_model="distilbert-base-cased-distilled-squad",
                 embedding_model="intfloat/e5-base-v2"):
        """Inicializa pipeline de QA con DistilBERT + FAISS"""

        # Cargar índice FAISS y chunks
        print("📂 Cargando índice y chunks...")
        self.indice = faiss.read_index(indice_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.embeddings_originales = np.load(embeddings_path)
        print(f"✅ Índice cargado: {self.indice.ntotal} vectores")

        # Cargar modelo de embeddings
        print("Cargando modelo de embeddings...")
        self.modelo_embeddings = SentenceTransformer(embedding_model)
        print("✅ Modelo embeddings listo")

        # Cargar modelo QA
        print("Cargando modelo DistilBERT QA...")
        self.qa_pipeline = pipeline("document-question-answering", model=qa_model)
        print("✅ Modelo QA listo")

    def buscar_chunks_hibrido(self, pregunta, top_k=3):
        """Búsqueda híbrida: metadatos + embeddings"""
        pregunta_lower = pregunta.lower()

        # Detectar chunks por metadatos
        candidatos_indices = []
        for i, chunk in enumerate(self.chunks):
            cancion = chunk[0].lower()
            artista = chunk[1].lower()
            palabras_artista = artista.split()
            palabras_cancion = [p for p in cancion.split() if len(p) > 3]

            if any(p in pregunta_lower for p in palabras_artista) or \
                    any(p in pregunta_lower for p in palabras_cancion):
                candidatos_indices.append(i)

        if candidatos_indices:
            print(f"   🎯 Filtro por metadatos: {len(candidatos_indices)} chunks")

            sub_embeddings = self.embeddings_originales[candidatos_indices].copy().astype('float32')
            faiss.normalize_L2(sub_embeddings)

            embedding_pregunta = self.modelo_embeddings.encode(
                [pregunta], convert_to_numpy=True
            ).astype('float32')
            faiss.normalize_L2(embedding_pregunta)

            scores = sub_embeddings @ embedding_pregunta.T
            scores = scores.flatten()
            top_local = np.argsort(scores)[::-1][:top_k]

            resultados = []
            for rank_i in top_local:
                real_i = candidatos_indices[rank_i]
                chunk = self.chunks[real_i]
                resultados.append({
                    "cancion": chunk[0],
                    "artista": chunk[1],
                    "letra": chunk[2],
                    "score": float(scores[rank_i]),
                    "indice": real_i
                })
            return resultados

        # Fallback: FAISS global
        print("   ⚠️ Sin match por metadatos, usando FAISS global")
        embedding_pregunta = self.modelo_embeddings.encode(
            [pregunta], convert_to_numpy=True
        ).astype('float32')
        faiss.normalize_L2(embedding_pregunta)

        distancias, indices = self.indice.search(embedding_pregunta, top_k)

        resultados = []
        for dist, idx in zip(distancias[0], indices[0]):
            chunk = self.chunks[idx]
            resultados.append({
                "cancion": chunk[0],
                "artista": chunk[1],
                "letra": chunk[2],
                "score": float(1 - dist),
                "indice": int(idx)
            })
        return resultados

    def qa_con_faiss(self, pregunta_en, top_k=3):
        """QA con DistilBERT + FAISS"""
        # Buscar chunks relevantes
        resultados = self.buscar_chunks_hibrido(pregunta_en, top_k)

        # Crear contexto limpio
        contexto = "\n".join([
            f"{r['cancion']} - {r['artista']}\n{r['letra']}"
            for r in resultados
        ])[:1500]

        # Ejecutar QA
        resultado = self.qa_pipeline(
            question=pregunta_en,
            context=contexto
        )

        return resultado, resultados

    def responder(self, pregunta_en, top_k=10):
        """Ejecuta QA y muestra resultados formateados"""
        resultado, chunks_usados = self.qa_con_faiss(pregunta_en, top_k)

        print("\nQUESTION ANSWERING - DistilBERT + FAISS")
        print("=" * 60)

        for i, r in enumerate(chunks_usados):
            print(f"   [{i + 1}] {r['cancion']} - {r['artista']}")
            print(f"        {r['letra'][:80]}...")

        print(f"\n✅ Answer : {resultado['answer']}")
        print(f"   Confidence : {resultado['score']:.4f}")
        print("=" * 60)

        return resultado

# ==============================
# Analisis de sentimientos
# ==============================
class SentimentAnalyzer:
    def __init__(self):
        """Inicializa modelos de análisis de sentimiento"""
        print("Cargando modelos de sentimiento...")
        self.sentimiento_multi = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.sentimiento_en = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
        # Mapeo de labels
        self.mapa = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }
        print("✅ Modelos de sentimiento listos")

    def analizar_cancion(self, texto):
        """Analiza sentimiento de una canción"""
        texto_truncado = texto[:512]
        res_multi = self.sentimiento_multi(texto_truncado)[0]
        res_en = self.sentimiento_en(texto_truncado)[0]
        label_en = self.mapa.get(res_en["label"], res_en["label"])
        return {
            "multi": res_multi["label"],
            "multi_score": res_multi["score"],
            "roberta": label_en,
            "roberta_score": res_en["score"]
        }

    def analizar_canciones_mongo(self, mongo_uri, db_name, collection_name, n=15):
        """Analiza sentimiento de canciones desde MongoDB (MUESTRA ALEATORIA)"""
        from pymongo import MongoClient
        client = MongoClient(mongo_uri)
        db = client[db_name]
        col = db[collection_name]

        # Usar $sample para obtener documentos aleatorios
        canciones = list(col.aggregate([
            {"$sample": {"size": n}}
        ]))

        # Encabezado con formato fijo
        print(f"\n{'Canción':<50} {'Multi':<15} {'RoBERTa'}")
        print("-" * 100)

        resultados = []
        for c in canciones:
            sentiment = self.analizar_cancion(c["letra"])
            nombre = f"{c['titulo']} - {c['artista']}"[:47]

            # Formato exacto con anchos fijos
            multi_label = sentiment['multi']
            roberta_label = sentiment['roberta']
            roberta_score = sentiment['roberta_score']

            # Imprimir con formato alineado
            print(f"{nombre:<50} {multi_label:<15} {roberta_label} ({roberta_score:.2f})")

            resultados.append({
                "cancion": c["titulo"],
                "artista": c["artista"],
                "sentimiento": sentiment
            })

        return resultados

    def analizar_lista_canciones(self, canciones):
        """Analiza sentimiento de una lista de canciones"""
        # Encabezado con formato fijo
        print(f"\n{'Canción':<50} {'Multi':<15} {'RoBERTa'}")
        print("-" * 100)

        resultados = []
        for c in canciones:
            sentiment = self.analizar_cancion(c["letra"])
            nombre = f"{c['titulo']} - {c['artista']}"[:47]

            # Formato exacto con anchos fijos
            multi_label = sentiment['multi']
            roberta_label = sentiment['roberta']
            roberta_score = sentiment['roberta_score']

            # Imprimir con formato alineado
            print(f"{nombre:<50} {multi_label:<15} {roberta_label} ({roberta_score:.2f})")

            resultados.append({
                "cancion": c["titulo"],
                "artista": c["artista"],
                "sentimiento": sentiment
            })

        return resultados

    def comparar_con_openai(self, mongo_uri, db_name, collection_name, generar_con_openai, n=3):
        """Compara análisis de BERT vs OpenAI en las mismas canciones"""
        from pymongo import MongoClient
        client = MongoClient(mongo_uri)
        db = client[db_name]
        col = db[collection_name]

        # Obtener muestra aleatoria
        canciones = list(col.aggregate([
            {"$sample": {"size": n}},
            {"$project": {
                "_id": 0,
                "titulo": 1,
                "artista": 1,
                "letra": 1
            }}
        ]))

        resultados_comparacion = []

        for cancion in canciones:
            print(f"\n{'=' * 80}")
            print(f"🎵 {cancion['titulo']} - {cancion['artista']}")
            print(f"{'=' * 80}")

            # 1. Análisis con BERT
            sentiment_bert = self.analizar_cancion(cancion["letra"])
            print(f"\n📊 BERT Multi:   {sentiment_bert['multi']:<15} (score: {sentiment_bert['multi_score']:.2f})")
            print(f"📊 BERT RoBERTa: {sentiment_bert['roberta']:<15} (score: {sentiment_bert['roberta_score']:.2f})")

            # 2. Análisis con OpenAI
            contexto = cancion["letra"]
            pregunta = f"""
Analiza el sentimiento de esta canción.
Clasifica en:
- Positivo
- Negativo
- Neutro
Identifica emociones:
- alegría, tristeza, enojo, miedo, sorpresa
Da un score de 1 a 10.
Canción:
{cancion['titulo']} - {cancion['artista']}
"""
            respuesta_openai = generar_con_openai(contexto, pregunta)
            print(f"\n🤖 OPENAI:\n{respuesta_openai}")

            # Guardar para análisis posterior
            resultados_comparacion.append({
                "cancion": cancion["titulo"],
                "artista": cancion["artista"],
                "bert": sentiment_bert,
                "openai": respuesta_openai
            })

        return resultados_comparacion


from openai import OpenAI
import os

def generar_con_openai(contexto, pregunta):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
Analiza el sentimiento de esta canción.

{pregunta}

Letra:
{contexto}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# ==============================
# Traductor Refactorizado
# ==============================
from transformers import MarianMTModel, MarianTokenizer
from pymongo import MongoClient


class TraductorPipeline:
    def __init__(self, openai_func, mongo_uri, db_name, collection_name):
        """
        Inicializa el TraductorPipeline con conexión a MongoDB

        Args:
            openai_func: Función para llamadas a OpenAI
            mongo_uri: URI de conexión a MongoDB
            db_name: Nombre de la base de datos
            collection_name: Nombre de la colección
        """
        self.openai_func = openai_func

        # Conexión a MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = self.db[collection_name]

        # Modelos de traducción
        self.model_es_en = None
        self.tokenizer_es_en = None
        self.model_en_es = None
        self.tokenizer_en_es = None

    def cargar_modelos(self):
        """Carga los modelos de traducción (ES->EN y EN->ES)"""
        print("Cargando ES -> EN...")
        model_name_es_en = "Helsinki-NLP/opus-mt-es-en"
        self.tokenizer_es_en = MarianTokenizer.from_pretrained(model_name_es_en)
        self.model_es_en = MarianMTModel.from_pretrained(model_name_es_en)

        print("Cargando EN -> ES...")
        model_name_en_es = "Helsinki-NLP/opus-mt-en-es"
        self.tokenizer_en_es = MarianTokenizer.from_pretrained(model_name_en_es)
        self.model_en_es = MarianMTModel.from_pretrained(model_name_en_es)

        print("✅ Modelos listos.")

    def traducir_local_en_es(self, texto):
        """Traduce de EN a ES usando modelo local"""
        tokens = self.tokenizer_en_es(texto, return_tensors="pt", padding=True, truncation=True)
        gen = self.model_en_es.generate(**tokens)
        return self.tokenizer_en_es.decode(gen[0], skip_special_tokens=True)

    def traducir_openai(self, texto):
        """Traduce de EN a ES usando OpenAI"""
        prompt = f"""
Translate from English to Spanish.
Return ONLY the translation.

Text:
{texto}
"""
        return self.openai_func("", prompt)

    def get_documentos_aleatorios(self, n=5):
        """
        Obtiene n documentos aleatorios de MongoDB

        Args:
            n: Cantidad de documentos aleatorios a obtener

        Returns:
            Lista de documentos con campos: titulo, artista, letra
        """
        documentos = list(self.col.aggregate([
            {"$sample": {"size": n}},
            {"$project": {
                "_id": 0,
                "titulo": 1,
                "artista": 1,
                "letra": 1
            }}
        ]))
        return documentos

    def procesar_documento(self, doc, max_chars=200):
        """Procesa un documento individual"""
        texto = doc.get("letra", "")[:max_chars]

        if not texto:
            return None

        return {
            "titulo": doc.get("titulo", ""),
            "artista": doc.get("artista", ""),
            "original": texto,
            "traduccion_local": self.traducir_local_en_es(texto),
            "traduccion_openai": self.traducir_openai(texto)
        }

    def procesar_lista(self, docs=None, max_chars=200):
        if docs is None:
            print("📂 Obteniendo documentos aleatorios de MongoDB...")
            docs = self.get_documentos_aleatorios(n=5)

        if not docs:
            print("⚠️ No hay documentos para procesar")
            return []

        resultados = []

        for doc in docs:
            try:
                res = self.procesar_documento(doc, max_chars)

                if res is None:
                    continue

                print("\n" + "=" * 60)

                print(f"🎤 Artista: {res['artista']}")
                print(f"🎵 Canción: {res['titulo']}")

                print("\n📄 Letra original:")
                print(res["original"])

                print("\n🤖 Traducción (Modelo local):")
                print(res["traduccion_local"])

                print("\n🔵 Traducción (OpenAI):")
                print(res["traduccion_openai"])

                print("=" * 60)

                resultados.append(res)

            except Exception as e:
                print(f"⚠️ Error procesando documento: {e}")
                continue

        print(f"\n✅ Total procesadas: {len(resultados)}")

        return resultados


# ==============================
# Traductor Refactorizado
# ==============================

from transformers import pipeline
from pymongo import MongoClient


class ResumenPipeline:
    def __init__(self, openai_func, mongo_uri, db_name, collection_name):
        """
        Pipeline para resumir letras de canciones

        Args:
            openai_func: función para llamar a OpenAI
            mongo_uri: conexión MongoDB
            db_name: base de datos
            collection_name: colección
        """
        self.openai_func = openai_func

        # MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = self.db[collection_name]

        # Modelo
        self.model = None
        self.tokenizer = None

    # =========================
    # CARGAR MODELO
    # =========================
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    def cargar_modelo(self):
        print("Cargando Flan-T5 para resúmenes...")

        model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print("✅ Modelo listo.")

    # =========================
    # OBTENER DOC
    # =========================
    def get_documento_aleatorio(self):
        doc = list(self.col.aggregate([
            {"$sample": {"size": 1}},
            {"$project": {
                "_id": 0,
                "titulo": 1,
                "artista": 1,
                "letra": 1
            }}
        ]))

        return doc[0] if doc else None

    # =========================
    # RESUMEN LOCAL
    # =========================
    def resumir_local(self, texto):
        prompt = f"Summarize the theme of this song in 2 sentences:\n\n{texto}"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# RESUMEN OPENAI
# =========================

    def resumir_openai(self, texto):
        prompt = f"""
Resume la siguiente letra de canción en máximo 2 oraciones en español:

{texto}
"""
        return self.openai_func("", prompt)

    # =========================
    # PROCESAR 1 DOCUMENTO
    # =========================
    def procesar_documento(self, doc, max_chars=600):
        if not doc:
            return None

        texto = doc.get("letra", "")[:max_chars]

        if not texto:
            return None

        return {
            "titulo": doc.get("titulo", ""),
            "artista": doc.get("artista", ""),
            "original": texto,
            "resumen_local": self.resumir_local(texto),
            "resumen_openai": self.resumir_openai(texto)
        }

    # =========================
    # PIPELINE COMPLETO
    # =========================
    def ejecutar(self, doc=None, max_chars=600):
        if doc is None:
            print("📂 Obteniendo documento aleatorio...")
            doc = self.get_documento_aleatorio()

        if not doc:
            print("⚠️ No se encontró documento")
            return None

        res = self.procesar_documento(doc, max_chars)

        if res is None:
            print("⚠️ Error procesando documento")
            return None

        print("\n" + "=" * 60)
        print(f"🎤 Artista: {res['artista'].title()}")
        print(f"🎵 Canción: {res['titulo'].title()}")
        print("=" * 60)

        print("\n📄 Letra original:")
        print(res["original"])

        print("\n🤖 Resumen (Flan-T5):")
        print(res["resumen_local"])

        print("\n🧠 Resumen (OpenAI):")
        print(res["resumen_openai"])

        print("=" * 60)

        return res


# ==============================
# Agente Conversacional
# ==============================
class AgenteRAGConversacional:
    def __init__(self, indice_path, chunks_path, model_name="intfloat/e5-base-v2", openai_func=None):
        """
        Agente RAG con memoria conversacional + FAISS + logging
        """

        self.openai_func = openai_func

        # =========================
        # LOG FILE
        # =========================
        self.log_path = "conversaciones.csv"

        # =========================
        # CARGA DE ÍNDICE
        # =========================
        print("📂 Cargando índice y chunks...")

        self.indice = faiss.read_index(indice_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"✅ Índice cargado: {self.indice.ntotal} vectores")

        # =========================
        # MODELO EMBEDDINGS
        # =========================
        print("Cargando modelo embeddings...")
        self.modelo_embeddings = SentenceTransformer(model_name)
        print("✅ Modelo listo")

# =========================
# MEMORIA CONVERSACIONAL
# =========================

        self.historial = []

        # =========================
        # SYSTEM PROMPT
        # =========================
        self.system_prompt = """
Eres MúsicBot 🎵, un asistente experto en música.

FORMATO OBLIGATORIO:
- Usa saltos de línea
- Usa listas con guiones o números
- Nunca escribas todo en un solo párrafo
- Respuestas claras y legibles

REGLAS IMPORTANTES:
- Solo respondes usando las letras del contexto.
- Nunca inventes canciones o artistas.
- Si no sabes, dilo claramente.
- Mantén un tono amigable y conversacional.
- Puedes recomendar canciones del corpus.
- Si el contexto no es suficiente, recomienda las más similares.

Siempre respondes en español.
"""

    # ============================================================
    # BÚSQUEDA RAG
    # ============================================================
    def buscar_chunks(self, pregunta, top_k=5):
        embedding = self.modelo_embeddings.encode(
            [pregunta],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(embedding)

        distancias, indices = self.indice.search(embedding, top_k)

        resultados = []
        for dist, idx in zip(distancias[0], indices[0]):
            titulo, artista, texto = self.chunks[idx]

            resultados.append({
                "titulo": titulo,
                "artista": artista,
                "texto": texto,
                "score": float(1 - dist)
            })

        return resultados

    # ============================================================
    # HISTORIAL
    # ============================================================
    def _formatear_historial(self):
        return "\n".join(
            f"{h['role']}: {h['content']}"
            for h in self.historial[-5:]
        )

    # ============================================================
    # PROMPT
    # ============================================================
    def _crear_prompt(self, pregunta, contexto):
        historial = self._formatear_historial()

        return f"""
{self.system_prompt}

CONTEXTO MUSICAL:
{contexto}

HISTORIAL:
{historial}

USUARIO:
{pregunta}

Responde siguiendo las reglas.
"""

    # ============================================================
    # OPENAI CALL
    # ============================================================
    def _llamar_llm(self, prompt):
        if self.openai_func:
            return self.openai_func("", prompt)

        from openai import OpenAI
        client = OpenAI()

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )

        return res.choices[0].message.content

    # ============================================================
    # GUARDAR CSV
    # ============================================================
    def guardar_conversacion(self, usuario, respuesta, resultados):
        fila = {
            "timestamp": datetime.now().isoformat(),
            "usuario": usuario,
            "respuesta": respuesta,
            "canciones": ", ".join([r["titulo"] for r in resultados])
        }

        df = pd.DataFrame([fila])

        if not os.path.exists(self.log_path):
            df.to_csv(self.log_path, index=False)
        else:
            df.to_csv(self.log_path, mode="a", header=False, index=False)

    # ============================================================
    # RESPUESTA PRINCIPAL
    # ============================================================
    def responder(self, mensaje, top_k=5):

        # 1. guardar usuario en memoria
        self.historial.append({"role": "Usuario", "content": mensaje})

        # 2. retrieval
        resultados = self.buscar_chunks(mensaje, top_k)

        contexto = "\n\n".join([
            f"""
CANCION: {r['titulo']}
ARTISTA: {r['artista']}
LETRA: {r['texto']}
"""
            for r in resultados
        ])

        # 3. prompt
        prompt = self._crear_prompt(mensaje, contexto)

        # 4. generar respuesta
        respuesta = self._llamar_llm(prompt)

        # 5. guardar memoria
        self.historial.append({"role": "Asistente", "content": respuesta})

        # 6. guardar en CSV
        self.guardar_conversacion(mensaje, respuesta, resultados)

        return respuesta, resultados

    # ============================================================
    # UTILIDAD
    # ============================================================
    def limpiar_historial(self):
        self.historial = []
        print("🧹 Historial limpiado")


# ============================================================
# CHATBOT FORMATO
# ============================================================

import random

def analizar_5_canciones_random(corpus, sentiment_model):
    muestras = random.sample(corpus, 5)
    resultados = []

    for i, c in enumerate(muestras):
        pred = sentiment_model.analizar_cancion(c["letra"])

        resultado = f"""
🎧 Canción {i+1}
🎵 Título: {c.get('titulo','desconocido')}
💬 Sentimiento: {pred['roberta']}
📊 Confianza: {pred['roberta_score']:.2f}
"""
        resultados.append(resultado)

    return "\n".join(resultados)