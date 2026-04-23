# 🎵 CorpusBeat — Chatbot Musical Inteligente con RAG + Fine-Tuning
 
> Agente conversacional que responde preguntas sobre música, artistas y letras de canciones, combinando búsqueda semántica (RAG), análisis de sentimiento, resumen y traducción, sobre un corpus de más de 5,000 canciones en español e inglés.
 
---
 
## 📌 Descripción General
 
CorpusBeat es un chatbot musical de extremo a extremo desarrollado como proyecto final del curso de **Minería de Textos** en el Colegio Universitario de Cartago. El sistema integra tres capas tecnológicas:
 
1. **Pipeline RAG** (Retrieval-Augmented Generation): recuperación semántica de letras relevantes usando FAISS + embeddings de `sentence-transformers`.
2. **Clasificador Fine-Tuneado**: modelo Transformer especializado en clasificación de género/sentimiento sobre el corpus propio.
3. **Interfaz conversacional con Plotly Dash**: chatbot web con memoria de diálogo, personalidad definida y módulos de análisis bajo demanda.
El chatbot **nunca inventa información**: cada respuesta se fundamenta en letras reales del corpus almacenado en MongoDB.
 
---
 
## 🗂️ Estructura del Proyecto
 
```
proyecto3_chatbot_musical/
├── notebooks/
│   ├── 01_exploracion_corpus.ipynb       # Estadísticas y análisis del dataset
│   └── 02_rag_pipeline.ipynb             # Chunking, embeddings, FAISS y generador
│
├── app/
│   └── chatbot_app.py                    # Aplicación Plotly Dash (punto de entrada)
│
├── src/
│   ├── rag_utils.py                      # Clases RAG, chunking, sentimiento, resumen, traducción
│   ├── mongoDB.py                        # Conector MongoDB reutilizable
│   └── rag_bot.py                        # Bot standalone (carga FAISS + responde)
│
├── data/
│   └── embeddings_cache/
│       ├── indice_parrafos.faiss         # Índice vectorial pre-construido
│       └── chunks_parrafos.pkl           # Chunks indexados con metadatos
│
├── models/                               # Modelo fine-tuneado guardado
├── resultados/                           # Métricas, matrices de confusión, logs
├── .env                                  # Variables de entorno (no incluido en el repo)
├── README.md
└── USO_DE_IA.md
```
 
---
 
## ⚙️ Arquitectura del Sistema
 
```
Usuario
   │
   ▼
┌─────────────────────────────────────────┐
│           Plotly Dash (UI)              │
│     Router de intención (NLU)           │
└────────┬────────────────────────────────┘
         │
   ┌─────▼──────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
   │  RAG Chat  │   │  Sentimiento  │   │    Resumen   │   │  Traducción  │
   │  (FAISS)   │   │  (Transformer)│   │  (OpenAI/T5) │   │  (Helsinki)  │
   └─────┬──────┘   └───────────────┘   └──────────────┘   └──────────────┘
         │
   ┌─────▼──────────────────────────────────────┐
   │  AgenteRAGConversacional                   │
   │  - Memoria de los últimos N turnos          │
   │  - Prompt de sistema con personalidad       │
   │  - Búsqueda semántica + BM25 híbrida        │
   │  - Generación con OpenAI GPT-4o-mini        │
   └─────┬──────────────────────────────────────┘
         │
   ┌─────▼──────────────────────────────────────┐
   │  MongoDB  ◄──►  FAISS Index (e5-base-v2)   │
   │  (corpus de canciones)                      │
   └────────────────────────────────────────────┘
```
 
---
 
## 🔍 Estrategias de Chunking
 
El módulo `MusicChunker` implementa y compara tres estrategias para fragmentar las letras antes de indexarlas:
 
| Estrategia | Descripción | Tamaño aprox. |
|---|---|---|
| **Fijo** | Ventana de N caracteres con overlap | ~400 chars |
| **Por oraciones** | Agrupa N oraciones con solapamiento | Variable |
| **Por párrafos** | Divide por doble salto de línea (estrofas) | Variable |
 
La estrategia por párrafos fue seleccionada como principal al preservar mejor la coherencia semántica de las estrofas.
 
---
 
## 🧠 Módulos del Sistema
 
### `AgenteRAGConversacional`
Clase central del chatbot. Mantiene historial de conversación (últimos N turnos), construye el prompt de sistema con personalidad y llama al LLM (OpenAI) con contexto recuperado de FAISS.
 
### `RAGPipeline`
Pipeline completo: carga el índice FAISS, genera embeddings con `intfloat/e5-base-v2`, realiza búsqueda híbrida (semántica + BM25) y genera respuesta via OpenAI.
 
### `SentimentAnalyzer`
Analiza el sentimiento de letras usando el modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student`. Permite comparar el resultado del modelo fine-tuneado vs. un análisis baseline con OpenAI.
 
### `ResumenPipeline`
Genera resúmenes de canciones del corpus usando `facebook/bart-large-cnn` de forma local o GPT-4o-mini como alternativa.
 
### `TraductorPipeline`
Traduce letras de inglés a español y viceversa usando `Helsinki-NLP/opus-mt-en-es` localmente, con fallback a OpenAI.
 
### `MongoReader`
Conector MongoDB simple con validación de conexión real, lectura del corpus completo como `DataFrame`.
 
---
 
## 🚀 Instalación y Uso
 
### 1. Clonar el repositorio
 
```bash
git clone https://github.com/tu-usuario/proyecto3-chatbot-musical.git
cd proyecto3-chatbot-musical
```
 
### 2. Instalar dependencias
 
```bash
pip install -r requirements.txt
```
 
### 3. Configurar variables de entorno
 
Crear un archivo `.env` en `src/`:
 
```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=musica
COLLECTION_NAME=canciones
OPENAI_API_KEY=sk-...
```
 
> ⚠️ El chatbot puede funcionar **sin API de OpenAI** usando `google/flan-t5-base` como generador local. La API es opcional.
 
### 4. Pre-construir el índice FAISS (solo la primera vez)
 
```bash
python src/rag_utils.py
```
 
Esto genera `indice_parrafos.faiss` y `chunks_parrafos.pkl` en `data/embeddings_cache/`.
 
### 5. Lanzar el chatbot
 
```bash
python app/chatbot_app.py
```
 
Abre `http://127.0.0.1:8050` en tu navegador.
 
---
 
## 📦 Dependencias Principales
 
| Librería | Uso |
|---|---|
| `sentence-transformers` | Generación de embeddings (`intfloat/e5-base-v2`) |
| `faiss-cpu` | Índice vectorial para búsqueda semántica |
| `transformers` | Modelos Transformer para sentimiento, resumen y traducción |
| `openai` | Generador de lenguaje (GPT-4o-mini) |
| `pymongo` | Conexión al corpus en MongoDB |
| `plotly-dash` | Interfaz web del chatbot |
| `pandas` / `numpy` | Manipulación de datos |
| `python-dotenv` | Gestión de variables de entorno |
| `scikit-learn` | Métricas de evaluación del clasificador |
 
---
 
## 📊 Evaluación
 
### Pipeline RAG
 
- Comparación de tres estrategias de chunking (tamaño promedio, coherencia semántica).
- Evaluación cualitativa de respuestas con y sin contexto RAG.
- Top-K recuperados: 5 chunks por defecto, ajustable.
### Clasificador Fine-Tuneado
 
Métricas sobre el conjunto de test (70/15/15 split, seed fijo):
 
| Métrica | Valor |
|---|---|
| Accuracy | *ver `resultados/metricas.json`* |
| F1 Macro | *ver `resultados/metricas.json`* |
| Baseline zero-shot | *comparado en notebook 03* |
 
---
 
## 💬 Ejemplos de Conversación
 
```
Usuario: ¿Qué canciones hablan de desamor?
MúsicBot: Encontré estas canciones sobre desamor en el corpus:
  1. "Ojalá" — Silvio Rodríguez
  2. "Sin Ti" — [Artista]
  ¿Te gustaría explorar alguna en detalle?
 
Usuario: Háblame más de la primera.
MúsicBot: "Ojalá" pertenece al género Nueva Trova, publicada en los 80...
  (MúsicBot recuerda el contexto anterior gracias a la memoria conversacional)
 
Usuario: Analiza el sentimiento de esa canción.
MúsicBot: El análisis de sentimiento arroja: Negativo (tristeza/nostalgia) con 87% de confianza.
```
 
---
 
## 🗃️ Corpus
 
- **Tamaño:** 5,000–10,000 canciones
- **Géneros:** Rock, Pop, Hip-Hop, Reggaetón, Baladas (mínimo 3)
- **Campos por canción:** título, artista, género, año, letra completa
- **Almacenamiento:** MongoDB (principal) + CSV de respaldo
> El corpus es el mismo utilizado durante los Proyectos 1 (POS Tagging) y 2 (Word2Vec / BETO) del semestre.
 
---
 
## 📁 Archivos de Resultados
 
```
resultados/
├── metricas.json           # Accuracy, F1, matriz de confusión
├── conversaciones_prueba/  # 10+ conversaciones documentadas (con y sin RAG)
└── confusion_matrix.png    # Visualización de la matriz de confusión
```
 
---
 
## 📝 Notas Técnicas
 
- Los embeddings del corpus se **cachean en disco** y se reutilizan entre ejecuciones, evitando recalcularlos cada vez.
- El sistema usa **búsqueda híbrida**: FAISS (semántica) + BM25 (léxica), combinando lo mejor de ambas.
- El generador está encapsulado — cambiar entre OpenAI y Flan-T5 basta con ajustar una variable de entorno.
- La aplicación Dash arranca con un **único comando** y no requiere instalación adicional más allá de `pip install`.
---
 
## 👥 Autores
 
Proyecto desarrollado para el curso **Minería de Textos**  
Colegio Universitario de Cartago · 2025  
Profesor: Osvaldo González Chaves
 
---
 
## 📄 Licencia
 
Este proyecto es de uso académico. Consulta el archivo `USO_DE_IA.md` para ver la política de uso de herramientas de inteligencia artificial durante el desarrollo.
 
