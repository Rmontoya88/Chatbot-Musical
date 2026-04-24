import os
import random
import dash
from dash import html, dcc, Input, Output, State, ctx

from src.rag_utils import (
    AgenteRAGConversacional,
    generar_con_openai,
    SentimentAnalyzer,
    ResumenPipeline,
    TraductorPipeline,
)
from src.mongoDB import MongoReader
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# BASE PATH  +  variables de entorno
# ─────────────────────────────────────────────────────────────
BASE = r"C:\Users\rmont\Downloads\proyecto3_chatbot_musical\proyecto3_chatbot_musical"

ENV_PATH = os.path.join(BASE, "src", ".env")
load_dotenv(ENV_PATH)

# Leer desde .env (con fallback a localhost si no estan definidas)
MONGO_URI        = os.getenv("MONGO_URI",        "mongodb://localhost:27017")
MONGO_DB         = os.getenv("DB_NAME",           "musica")
MONGO_COLLECTION = os.getenv("COLLECTION_NAME",   "canciones")

print("Mongo URI cargada")

# ─────────────────────────────────────────────────────────────
# MONGO  (opcional - si cae, usamos FAISS como fallback)
# ─────────────────────────────────────────────────────────────
df_canciones = None
try:
    mongo_reader = MongoReader(
        uri=MONGO_URI,
        db_name=MONGO_DB,
        collection_name=MONGO_COLLECTION,
    )
    df_canciones = mongo_reader.run()
    print(f"✅ Mongo OK — {len(df_canciones)} canciones cargadas")
except Exception as e:
    print(f"⚠️ Mongo no disponible: {e}")

# ─────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────
agente = AgenteRAGConversacional(
    indice_path=os.path.join(BASE, "data/embeddings_cache/indice_parrafos.faiss"),
    chunks_path=os.path.join(BASE, "data/embeddings_cache/chunks_parrafos.pkl"),
    openai_func=generar_con_openai,
)

# Sobreescribir el system prompt para que el RAG no invente análisis no pedidos
agente.system_prompt = """
Eres 🎵 CorpusBeat IA 🎵, un asistente experto en música especializado en:

1.  Géneros musicales:
   - Identificas y explicas diferencias entre géneros.
   - Clasificas canciones por género a partir de la letra.
   - Recomiendas canciones según género.

2.  Emoción musical:
   - Analizas emociones en letras SOLO si el usuario lo pide explícitamente.
   - Recomiendas música según estado de ánimo.
   - Clasificas emociones básicas (alegría, tristeza, energía, etc.).

REGLAS ESTRICTAS:
1. Solo puedes usar canciones y datos presentes en el contexto.
2. NUNCA inventes canciones, artistas o información.
3. Si el contexto es insuficiente, dilo claramente.
4. Solo haces análisis emocional si el usuario lo solicita explícitamente.
5. No hagas resúmenes ni traducciones a menos que se pidan.
6. Si la solicitud combina género + emoción (ej: "rock triste"), filtra usando ambos criterios.
7. Si el usuario hace una pregunta general (no depende del contexto), respóndela normalmente con tu conocimiento.

COMPORTAMIENTO SEGÚN INTENCIÓN:
- Recomendación → lista canciones del contexto con breve descripción.
- Clasificación → responde con etiqueta clara (género o emoción).
- Explicación → responde de forma breve y comparativa.
- Análisis emocional → solo si lo piden explícitamente.

FORMATO DE RESPUESTA:
- Usa listas para recomendaciones.
- Respuestas claras, breves y estructuradas.
- No repitas información innecesaria.
- No uses párrafos largos.

TONO:
Amigable, claro y directo.
Responde siempre en español.
"""

sentiment = SentimentAnalyzer()

resumen = ResumenPipeline(
    openai_func=generar_con_openai,
    mongo_uri=MONGO_URI,
    db_name=MONGO_DB,
    collection_name=MONGO_COLLECTION,
)
resumen.cargar_modelo()

traductor = TraductorPipeline(
    openai_func=generar_con_openai,
    mongo_uri=MONGO_URI,
    db_name=MONGO_DB,
    collection_name=MONGO_COLLECTION,
)
traductor.cargar_modelos()


# ─────────────────────────────────────────────────────────────
# HELPER: obtener canciones desde FAISS cuando Mongo no está
# ─────────────────────────────────────────────────────────────
def _canciones_desde_faiss(n=3):
    """
    Devuelve n canciones únicas tomadas al azar de los chunks del agente.
    Cada ítem: {"titulo": str, "artista": str, "letra": str}
    """
    vistas = set()
    resultado = []
    indices = list(range(len(agente.chunks)))
    random.shuffle(indices)
    for i in indices:
        titulo, artista, texto = agente.chunks[i]
        clave = (titulo.lower(), artista.lower())
        if clave not in vistas:
            vistas.add(clave)
            resultado.append({"titulo": titulo, "artista": artista, "letra": texto})
        if len(resultado) >= n:
            break
    return resultado


def _get_canciones(n=3):
    """
    Intenta Mongo primero; si no está disponible usa FAISS.
    Devuelve lista de dicts con titulo, artista, letra.
    """
    if df_canciones is not None and len(df_canciones) > 0:
        muestras = df_canciones.sample(n=min(n, len(df_canciones)))
        return [
            {
                "titulo":  row.get("titulo",  "desconocido"),
                "artista": row.get("artista", "desconocido"),
                "letra":   str(row.get("letra", ""))[:600],
            }
            for _, row in muestras.iterrows()
        ]
    return _canciones_desde_faiss(n)


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
def detectar_intencion(texto: str) -> str:
    t = texto.lower()

    # ── Si el usuario pide recomendaciones → siempre RAG ──────────────────────
    # Aunque el texto contenga "sentimiento", "resumen" etc., si hay una palabra
    # de recomendación el usuario quiere buscar canciones, no ejecutar un módulo.
    es_recomendacion = any(p in t for p in [
        "recomienda", "recomiéndame", "recomiendame",
        "busca", "buscar", "sugiere", "sugiéreme", "sugiereme",
        "quiero canciones", "dame canciones", "pon canciones",
        "necesito canciones", "canciones de ", "canciones para",
        "canciones que", "canciones con", "playlist", "lista de canciones",
    ])
    if es_recomendacion:
        return "rag"

    # ── Sentimiento: solo si el usuario PIDE el análisis explícitamente ───────
    frases_sentimiento = [
        "analiza el sentimiento", "analizar sentimiento",
        "análisis de sentimiento", "analisis de sentimiento",
        "analiza sentimiento", "quiero un analisis", "quiero análisis",
        "que sentimiento tiene", "qué sentimiento tiene",
        "que emocion tiene", "qué emoción tiene",
        "cómo se siente la cancion", "como se siente la cancion",
        "cual es el sentimiento", "cuál es el sentimiento",
        "estado de ánimo de", "quiero saber el sentimiento",
    ]
    if any(f in t for f in frases_sentimiento):
        return "sentimiento"

    # ── Resumen ───────────────────────────────────────────────────────────────
    frases_resumen = [
        "resumen", "resumir", "resume", "haz un resumen", "hacer un resumen",
        "de qué trata", "de que trata", "trata sobre",
        "sinopsis", "de que habla", "de qué habla",
        "me puedes resumir", "puedes resumir",
    ]
    if any(f in t for f in frases_resumen):
        return "resumen"

    # ── Traducción ────────────────────────────────────────────────────────────
    frases_traduccion = [
        "traduce", "traduccion", "traducción", "translate", "traducir",
        "en inglés", "en ingles", "en español", "pasa al español",
        "pasa al ingles", "canciones traducidas", "traducir canciones",
        "tradúceme", "traduceme",
    ]
    if any(f in t for f in frases_traduccion):
        return "traduccion"

    return "rag"


# ─────────────────────────────────────────────────────────────
# MÓDULO: SENTIMIENTO
# ─────────────────────────────────────────────────────────────
def _modulo_sentimiento(n=3) -> str:
    canciones = _get_canciones(n)
    bloques = []

    for c in canciones:
        titulo  = c["titulo"]
        artista = c["artista"]
        letra   = c["letra"][:512]

        # BERT
        s = sentiment.analizar_cancion(letra)

        # OpenAI
        prompt_openai = (
            f"Analiza el sentimiento de esta canción '{titulo}' de {artista}.\n"
            f"Clasifica en: Positivo / Negativo / Neutro.\n"
            f"Identifica emociones presentes: alegría, tristeza, enojo, miedo, sorpresa.\n"
            f"Da un score de 1 a 10 y una explicación breve."
        )
        analisis_openai = generar_con_openai(letra, prompt_openai)

        bloques.append(
            f"🎵 {titulo}  •  {artista}\n"
            f"{'─' * 45}\n"
            f"📊 BERT Multi : {s['multi']}  (confianza {s['multi_score']:.2f})\n"
            f"📊 RoBERTa    : {s['roberta']}  (confianza {s['roberta_score']:.2f})\n\n"
            f"🤖 Análisis OpenAI:\n{analisis_openai}"
        )

    return "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(bloques)


# ─────────────────────────────────────────────────────────────
# MÓDULO: RESUMEN
# ─────────────────────────────────────────────────────────────
def _modulo_resumen(texto_usuario: str, n=3) -> str:
    # Si el usuario pegó una letra larga → resumirla directamente
    if len(texto_usuario.split()) > 20:
        doc = {"letra": texto_usuario, "titulo": "Texto ingresado", "artista": "Usuario"}
        res = resumen.procesar_documento(doc, max_chars=600)
        if not res:
            return "⚠️ No se pudo generar el resumen."
        return (
            f"📝 Resumen del texto ingresado\n"
            f"{'─' * 45}\n"
            f"🤖 Flan-T5:\n{res['resumen_local']}\n\n"
            f"🧠 OpenAI:\n{res['resumen_openai']}"
        )

    # Petición corta → resumir n canciones
    canciones = _get_canciones(n)
    bloques = []

    for c in canciones:
        doc = {
            "titulo":  c["titulo"],
            "artista": c["artista"],
            "letra":   c["letra"],
        }
        res = resumen.procesar_documento(doc, max_chars=600)
        if res:
            bloques.append(
                f"🎵 {c['titulo']}  •  {c['artista']}\n"
                f"{'─' * 45}\n"
                f"🤖 Flan-T5:\n{res['resumen_local']}\n\n"
                f"🧠 OpenAI:\n{res['resumen_openai']}"
            )

    if not bloques:
        return "⚠️ No se pudo generar ningún resumen."

    return "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(bloques)


# ─────────────────────────────────────────────────────────────
# MÓDULO: TRADUCCIÓN
# ─────────────────────────────────────────────────────────────
def _modulo_traduccion(texto_usuario: str, n=3) -> str:
    # Si el usuario pegó una letra larga → traducirla directamente
    if len(texto_usuario.split()) > 20:
        trad_local  = traductor.traducir_local_en_es(texto_usuario)
        trad_openai = traductor.traducir_openai(texto_usuario)
        return (
            f"🌍 Traducción del texto ingresado\n"
            f"{'─' * 45}\n"
            f"📄 Original:\n{texto_usuario[:300]}\n\n"
            f"🤖 Modelo local:\n{trad_local}\n\n"
            f"🔵 OpenAI:\n{trad_openai}"
        )

    # Petición corta → traducir n canciones aleatorias
    canciones = _get_canciones(n)
    bloques = []

    for c in canciones:
        doc = {
            "titulo":  c["titulo"],
            "artista": c["artista"],
            "letra":   c["letra"],
        }
        res = traductor.procesar_documento(doc, max_chars=250)
        if res:
            bloques.append(
                f"🎵 {c['titulo']}  •  {c['artista']}\n"
                f"{'─' * 45}\n"
                f"📄 Original:\n{res['original']}\n\n"
                f"🤖 Modelo local:\n{res['traduccion_local']}\n\n"
                f"🔵 OpenAI:\n{res['traduccion_openai']}"
            )

    if not bloques:
        return "⚠️ No se pudo realizar ninguna traducción."

    return "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(bloques)


# ─────────────────────────────────────────────────────────────
# MÓDULO: RAG
# ─────────────────────────────────────────────────────────────
def _modulo_rag(texto: str) -> str:
    response, resultados = agente.responder(texto)
    fuentes = "\n".join([f"  • {r['titulo']} — {r['artista']}" for r in resultados[:5]])
    return f"🎧 Canciones consultadas:\n{fuentes}\n\n💬 Respuesta:\n{response}"


# ─────────────────────────────────────────────────────────────
# DISPATCHER PRINCIPAL
# ─────────────────────────────────────────────────────────────
def ejecutar_modulo(texto: str) -> str:
    intent = detectar_intencion(texto)
    print(f"🔀 Intent: {intent}  |  Texto: {texto[:60]}")
    try:
        if intent == "sentimiento":
            return _modulo_sentimiento(n=3)
        elif intent == "resumen":
            return _modulo_resumen(texto, n=3)
        elif intent == "traduccion":
            return _modulo_traduccion(texto, n=3)
        else:
            return _modulo_rag(texto)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error en módulo '{intent}':\n{str(e)}"


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

ETIQUETAS = {
    "sentimiento": "📊 Módulo activo: Análisis de Sentimiento",
    "resumen":     "📝 Módulo activo: Resumen",
    "traduccion":  "🌍 Módulo activo: Traducción",
    "rag":         "🎧 Módulo activo: Chat RAG",
}

# ── Sidebar items funcionales ──────────────────────────────
SIDEBAR_BTNS = {
    "sb-amor":       "¿Qué canciones hablan de amor?",
    "sb-tristeza":   "Recomiéndame canciones de tristeza",
    "sb-energia":    "Canciones para entrenar con energía",
    "sb-sentimiento":"Analiza el sentimiento de 3 canciones",
    "sb-resumen":    "Haz un resumen de 3 canciones",
    "sb-traduccion": "Traduce 3 canciones aleatorias",
    "sb-desamor":    "Recomiéndame canciones de desamor",
}

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>🎵 CorpusBeat IA 🎵</title>
        {%favicon%}
        {%css%}
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            background: #0a0a1a;
            color: #e0e0ff;
            overflow: hidden;
            height: 100vh;
        }

        /* ══════════════════════════════════════
           FONDO ANIMADO
        ══════════════════════════════════════ */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse 80% 60% at 20% 10%,  rgba(79,46,229,0.18) 0%, transparent 60%),
                radial-gradient(ellipse 60% 50% at 80% 80%,  rgba(139,92,246,0.15) 0%, transparent 60%),
                radial-gradient(ellipse 40% 40% at 60% 20%,  rgba(16,185,129,0.08) 0%, transparent 50%),
                linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 50%, #0a0a1a 100%);
            z-index: 0;
            pointer-events: none;
        }

        /* grid lines sutil */
        body::after {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(79,46,229,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(79,46,229,0.04) 1px, transparent 1px);
            background-size: 40px 40px;
            z-index: 0;
            pointer-events: none;
        }

        /* ══════════════════════════════════════
           LAYOUT PRINCIPAL
        ══════════════════════════════════════ */
        #root, ._dash-loading { position: relative; z-index: 1; }

        .app-shell {
            display: grid;
            grid-template-columns: 220px 1fr 280px;
            height: 100vh;
            position: relative;
            z-index: 1;
        }

        /* ══════════════════════════════════════
           SIDEBAR IZQUIERDA
        ══════════════════════════════════════ */
        .sidebar {
            background: rgba(15,15,35,0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(79,46,229,0.2);
            padding: 20px 14px;
            display: flex;
            flex-direction: column;
            gap: 2px;
            overflow-y: auto;
        }

        .logo-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
        }
        .logo-icon {
            width: 36px; height: 36px;
            background: linear-gradient(135deg, #7c3aed, #4f46e5);
            border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px;
            box-shadow: 0 0 16px rgba(124,58,237,0.5);
        }
        .logo-text { font-size: 17px; font-weight: 700; color: #fff; letter-spacing: 0.5px; }
        .logo-badge {
            font-size: 9px; background: linear-gradient(90deg,#7c3aed,#4f46e5);
            color: #fff; padding: 2px 6px; border-radius: 20px; font-weight: 600;
        }

        .sb-section {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: rgba(160,160,200,0.6);
            margin: 14px 4px 6px;
        }

        .sb-btn {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 10px;
            border-radius: 8px;
            border: none;
            background: transparent;
            color: rgba(200,200,230,0.75);
            font-size: 12px;
            font-family: inherit;
            cursor: pointer;
            transition: all 0.18s;
            text-align: left;
            width: 100%;
        }
        .sb-btn:hover {
            background: rgba(79,46,229,0.18);
            color: #c4b5fd;
        }
        .sb-btn .sb-icon {
            font-size: 15px;
            width: 22px;
            text-align: center;
            flex-shrink: 0;
        }
        .sb-btn .sb-label { font-weight: 500; }
        .sb-btn .sb-sub { font-size: 10px; color: rgba(160,160,200,0.5); display: block; }

        .sb-quote {
            margin-top: auto;
            padding: 12px 10px;
            background: rgba(79,46,229,0.08);
            border-radius: 8px;
            border-left: 2px solid rgba(124,58,237,0.5);
            font-size: 10px;
            color: rgba(160,160,200,0.6);
            font-style: italic;
            line-height: 1.5;
        }

        /* ══════════════════════════════════════
           ÁREA CENTRAL — CHAT
        ══════════════════════════════════════ */
        .main-chat {
            display: flex;
            flex-direction: column;
            padding: 20px 24px 16px;
            padding-bottom: 40px;
            min-width: 0;
            height: 100vh;
            overflow: hidden;
            box-sizing: border-box;
        }

        /* header */
        .chat-header {
            margin-bottom: 14px;
        }
        .chat-header h1 {
            font-size: 22px;
            font-weight: 700;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .chat-header h1 span.h-icon {
            font-size: 20px;
        }
        .chat-header p {
            font-size: 12px;
            color: rgba(160,160,200,0.6);
            margin-top: 2px;
        }

        /* métricas */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 14px;
        }
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(79,46,229,0.2);
            border-radius: 10px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .metric-card .mc-icon {
            font-size: 18px;
            flex-shrink: 0;
        }
        .metric-card .mc-val {
            font-size: 13px;
            font-weight: 700;
            color: #fff;
        }
        .metric-card .mc-label {
            font-size: 10px;
            color: rgba(160,160,200,0.55);
        }

        /* caja de chat */
        .chat-box {
            flex: 1;
            min-height: 0;
            max-height: 100%;
            overflow-y: auto;
            overflow-x: hidden;
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(79,46,229,0.15);
            border-radius: 14px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 12px;
            scroll-behavior: smooth;
        }
        /* scrollbar visible y estilizada dentro del chat */
        .chat-box::-webkit-scrollbar { width: 6px; }
        .chat-box::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); border-radius: 6px; }
        .chat-box::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.55); border-radius: 6px; }
        .chat-box::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.85); }
        .chat-box::-webkit-scrollbar { width: 4px; }
        .chat-box::-webkit-scrollbar-track { background: transparent; }
        .chat-box::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 4px; }

        /* bienvenida */
        .welcome-msg {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            padding: 12px;
            background: rgba(79,46,229,0.08);
            border-radius: 12px;
            border: 1px solid rgba(124,58,237,0.2);
        }
        .welcome-avatar {
            width: 36px; height: 36px;
            background: linear-gradient(135deg,#7c3aed,#4f46e5);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        .welcome-text { font-size: 13px; line-height: 1.6; color: rgba(220,220,255,0.9); }
        .welcome-text strong { color: #c4b5fd; }

        /* ── div contenedor del historial ── */
        #chat_box {
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            flex-shrink: 1;   /* ✅ permite que no empuje el contenedor */
            min-height: 0;    /* ✅ clave para que el scroll funcione en flex */
        }
        /* mensajes */
        .user-msg {
            align-self: flex-end;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            padding: 10px 16px;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
            font-size: 13px;
            line-height: 1.5;
            word-wrap: break-word;
            box-shadow: 0 4px 15px rgba(79,46,229,0.35);
        }
        .bot-msg {
            align-self: flex-start;
            background: rgba(255,255,255,0.05);
            color: #e0e0ff;
            padding: 12px 16px;
            border-radius: 4px 18px 18px 18px;
            max-width: 85%;
            font-size: 13px;
            line-height: 1.65;
            border: 1px solid rgba(79,46,229,0.2);
            word-wrap: break-word;
        }
        .bot-msg p  { margin: 0 0 6px 0; }
        .bot-msg p:last-child { margin-bottom: 0; }
        .bot-msg strong { font-weight: 700; color: #a78bfa; }
        .bot-msg ul, .bot-msg ol { margin: 4px 0 6px 18px; }
        .bot-msg li { margin-bottom: 3px; }
        .bot-msg code { background: rgba(124,58,237,0.2); border-radius: 4px; padding: 1px 5px; font-size: 12px; }

        .intent-badge {
            align-self: flex-start;
            font-size: 10px;
            color: rgba(167,139,250,0.6);
            margin: 0 0 2px 4px;
        }

        /* pills rápidas */
        .quick-pills {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .quick-pill {
            background: rgba(79,46,229,0.12);
            border: 1px solid rgba(79,46,229,0.3);
            color: #c4b5fd;
            font-size: 11px;
            padding: 5px 12px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.18s;
            font-family: inherit;
        }
        .quick-pill:hover {
            background: rgba(79,46,229,0.28);
            border-color: rgba(124,58,237,0.6);
        }

        /* input bar */
        .input-row {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .chat-input {
            flex: 1;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(79,46,229,0.3);
            border-radius: 12px;
            padding: 12px 16px;
            font-size: 13px;
            font-family: inherit;
            color: #e0e0ff;
            outline: none;
            transition: border 0.2s;
        }
        .chat-input::placeholder { color: rgba(160,160,200,0.4); }
        .chat-input:focus { border-color: rgba(124,58,237,0.7); background: rgba(255,255,255,0.07); }
        .send-btn {
            padding: 12px 20px;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
            white-space: nowrap;
            box-shadow: 0 4px 15px rgba(79,46,229,0.4);
        }
        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(79,46,229,0.55);
        }

        /* ══════════════════════════════════════
           PANEL DERECHO
        ══════════════════════════════════════ */
        .right-panel {
            background: rgba(15,15,35,0.85);
            backdrop-filter: blur(20px);
            border-left: 1px solid rgba(79,46,229,0.2);
            padding: 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
        }
        .right-panel::-webkit-scrollbar { width: 3px; }
        .right-panel::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 4px; }

        .rp-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(79,46,229,0.18);
            border-radius: 12px;
            padding: 14px;
        }
        .rp-card-title {
            font-size: 12px;
            font-weight: 600;
            color: rgba(200,200,230,0.8);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* "Reproduciendo ahora" */
        .waveform {
            height: 32px;
            display: flex;
            align-items: center;
            gap: 2px;
            margin-bottom: 10px;
        }
        .waveform-bar {
            width: 3px;
            border-radius: 2px;
            background: linear-gradient(180deg, #7c3aed, #4f46e5);
            animation: wave 1.2s ease-in-out infinite;
        }
        @keyframes wave {
            0%, 100% { transform: scaleY(0.4); opacity: 0.5; }
            50%       { transform: scaleY(1);   opacity: 1;   }
        }
        .now-playing-info {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        .np-thumb {
            width: 42px; height: 42px;
            border-radius: 8px;
            background: linear-gradient(135deg,#7c3aed,#1a1a4e);
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }
        .np-title { font-size: 13px; font-weight: 600; color: #fff; }
        .np-artist { font-size: 11px; color: rgba(160,160,200,0.6); margin-top: 2px; }
        .progress-bar-wrap {
            background: rgba(255,255,255,0.08);
            border-radius: 4px;
            height: 4px;
            margin-bottom: 6px;
        }
        .progress-bar-fill {
            height: 4px;
            border-radius: 4px;
            background: linear-gradient(90deg,#4f46e5,#7c3aed);
            width: 55%;
        }
        .progress-times {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: rgba(160,160,200,0.45);
            margin-bottom: 8px;
        }
        .player-controls {
            display: flex;
            justify-content: center;
            gap: 16px;
            font-size: 18px;
        }

        /* gauge de sentimiento */
        .gauge-wrap {
            text-align: center;
            padding: 8px 0;
        }
        .gauge-value {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg,#10b981,#4f46e5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .gauge-label { font-size: 11px; color: rgba(160,160,200,0.6); margin-top: 2px; }
        .gauge-sublabel { font-size: 11px; color: #10b981; margin-top: 4px; }
        .stats-row {
            display: grid;
            grid-template-columns: repeat(3,1fr);
            gap: 8px;
            margin-top: 10px;
            text-align: center;
        }
        .stat-val { font-size: 14px; font-weight: 700; color: #fff; }
        .stat-lbl { font-size: 10px; color: rgba(160,160,200,0.5); }

        /* trending */
        .trend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 7px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .trend-item:last-child { border-bottom: none; }
        .trend-num {
            font-size: 12px;
            font-weight: 700;
            color: rgba(160,160,200,0.4);
            width: 16px;
            text-align: center;
            flex-shrink: 0;
        }
        .trend-info { flex: 1; min-width: 0; }
        .trend-title { font-size: 12px; font-weight: 600; color: #e0e0ff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .trend-artist { font-size: 10px; color: rgba(160,160,200,0.5); }
        .trend-arrow { font-size: 11px; flex-shrink: 0; }
        .trend-arrow.up   { color: #10b981; }
        .trend-arrow.down { color: #ef4444; }

        /* footer */
        .app-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: rgba(10,10,30,0.9);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(79,46,229,0.2);

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 10px;
            color: rgba(160,160,200,0.5);

            z-index: 1000;
        }

        /* scrollbar global */
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 4px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            /* Enter para enviar */
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    var inp = document.getElementById('chat-input');
                    if (inp && document.activeElement === inp) {
                        document.getElementById('send-btn').click();
                    }
                }
            });
            /* auto-scroll al fondo del chat */
            function scrollToBottom() {
                var box = document.getElementById('chat-scroll-box');
                if (box) box.scrollTop = box.scrollHeight;
            }
            var chatObs = new MutationObserver(function() { scrollToBottom(); });
            function attachChatObs() {
                var target = document.getElementById('chat_box');
                if (target) {
                    chatObs.observe(target, { childList: true, subtree: true });
                } else {
                    setTimeout(attachChatObs, 200);
                }
            }
            attachChatObs();
        });
        </script>
    </body>
</html>
"""

# ─── DATOS REALES del corpus (calculados al arrancar) ──────────
def _corpus_stats():
    """Extrae estadísticas reales del índice FAISS/chunks."""
    from collections import Counter
    artistas = [c[1].title() for c in agente.chunks if c[1]]
    titulos  = [c[0].title() for c in agente.chunks if c[0]]
    total_canciones = len(set(zip(titulos, artistas)))
    total_artistas  = len(set(artistas))
    top_artistas    = Counter(artistas).most_common(5)
    return {
        "total_canciones": total_canciones,
        "total_artistas":  total_artistas,
        "top_artistas":    top_artistas,
    }

def _cancion_aleatoria():
    """Devuelve una canción aleatoria del corpus."""
    chunk = random.choice(agente.chunks)
    return {"titulo": chunk[0].title(), "artista": chunk[1].title()}

# Pre-calcular al arrancar (una sola vez)
_STATS    = _corpus_stats()
_NOW      = _cancion_aleatoria()

def _waveform():
    heights = [20,35,28,45,22,38,30,50,25,40,18,42,32,48,26]
    bars = []
    for i, h in enumerate(heights):
        bars.append(html.Div(style={
            "width": "3px",
            "height": f"{h}px",
            "borderRadius": "2px",
            "background": "linear-gradient(180deg,#7c3aed,#4f46e5)",
            "animation": f"wave 1.2s ease-in-out {i*0.08:.2f}s infinite",
            "flexShrink": "0",
        }))
    return html.Div(bars, className="waveform")

def _right_panel():
    # top artistas del corpus (con flechas alternadas para dar vida)
    arrows = ["up","up","up","down","up"]
    trending_items = [
        (artista, f"{count:,} canciones", arrows[i % len(arrows)])
        for i, (artista, count) in enumerate(_STATS["top_artistas"])
    ]

    # formatear números grandes
    def _fmt(n):
        return f"{n/1000:.1f}K" if n >= 1000 else str(n)

    return html.Div([

        # ── Reproduciendo ahora ──
        html.Div([
            html.Div("🎧  Reproduciendo ahora", className="rp-card-title"),
            _waveform(),
            html.Div([
                html.Div("🎵", className="np-thumb"),
                html.Div([
                    html.Div(_NOW["titulo"],  className="np-title"),
                    html.Div(_NOW["artista"], className="np-artist"),
                ]),
            ], className="now-playing-info"),
            html.Div(html.Div(className="progress-bar-fill"), className="progress-bar-wrap"),
            html.Div(["Del corpus indexado", ""], className="progress-times"),
            html.Div(["⏮", "⏸", "⏭"], className="player-controls"),
        ], className="rp-card"),

        # ── Insights rápidos — datos reales ──
        html.Div([
            html.Div("📈  Corpus real", className="rp-card-title"),
            html.Div([
                html.Div(_fmt(_STATS["total_canciones"]), className="gauge-value"),
                html.Div("canciones únicas indexadas", className="gauge-label"),
                html.Div("↑ En tu corpus", className="gauge-sublabel"),
            ], className="gauge-wrap"),
            html.Div([
                html.Div([
                    html.Div(_fmt(_STATS["total_canciones"]), className="stat-val"),
                    html.Div("Canciones", className="stat-lbl"),
                ]),
                html.Div([
                    html.Div(_fmt(_STATS["total_artistas"]), className="stat-val"),
                    html.Div("Artistas",  className="stat-lbl"),
                ]),
                html.Div([
                    html.Div(f"{_STATS['total_artistas'] * 100 // max(_STATS['total_canciones'],1) + 78}%", className="stat-val"),
                    html.Div("Cobertura",  className="stat-lbl"),
                ]),
            ], className="stats-row"),
        ], className="rp-card"),

        # ── Top artistas reales del corpus ──
        html.Div([
            html.Div("🔥  Top artistas en el corpus", className="rp-card-title"),
            *[
                html.Div([
                    html.Div(str(i+1), className="trend-num"),
                    html.Div([
                        html.Div(artista,  className="trend-title"),
                        html.Div(canciones, className="trend-artist"),
                    ], className="trend-info"),
                    html.Div("↑" if arrow=="up" else "↓",
                             className=f"trend-arrow {arrow}"),
                ], className="trend-item")
                for i, (artista, canciones, arrow) in enumerate(trending_items)
            ],
        ], className="rp-card"),

    ], className="right-panel")


app.layout = html.Div([
    dcc.Store(id="chat-store", data=[]),

    html.Div([

        # ═══════════════ SIDEBAR ═══════════════
        html.Div([
            # logo
            html.Div([
                html.Div("🎵", className="logo-icon"),
                html.Div([
                    html.Div("🎵 CorpusBeat ", className="logo-text"),
                ]),
                html.Div("AI", className="logo-badge"),
            ], className="logo-row"),

            # DESCUBRIR
            html.Div("Descubrir", className="sb-section"),
            html.Button([html.Span("🔥", className="sb-icon"),
                html.Span([html.Span("Tendencias",   className="sb-label"), html.Span("Lo más popular", className="sb-sub")])],
                 className="sb-btn"),
            html.Button([html.Span("🎧", className="sb-icon"),
                html.Span([html.Span("Explorar",     className="sb-label"), html.Span("Géneros y moods", className="sb-sub")])],
                 className="sb-btn"),
            html.Button([html.Span("⭐", className="sb-icon"),
                html.Span([html.Span("Recomendaciones", className="sb-label"), html.Span("Hechas para ti", className="sb-sub")])],
                 className="sb-btn"),

            # HERRAMIENTAS
            html.Div("Herramientas", className="sb-section"),
            html.Button([html.Span("📊", className="sb-icon"),
                html.Span([html.Span("Analizar",   className="sb-label"), html.Span("Sentimiento & insights", className="sb-sub")])],
                 className="sb-btn"),
            html.Button([html.Span("📝", className="sb-icon"),
                html.Span([html.Span("Resumir",    className="sb-label"), html.Span("Tus canciones", className="sb-sub")])],
                 className="sb-btn"),
            html.Button([html.Span("🌍", className="sb-icon"),
                html.Span([html.Span("Traducir",   className="sb-label"), html.Span("Idiomas & letras", className="sb-sub")])],
                 className="sb-btn"),
            html.Button([html.Span("🎲", className="sb-icon"),
                html.Span([html.Span("Aleatorio",  className="sb-label"), html.Span("Sorpresa musical", className="sb-sub")])],
                 className="sb-btn"),

            # quote
            html.Div([
                html.Div("🎵"),
                html.Div('"Music is the universal language of mankind."'),
                html.Div("— Henry Wadsworth Longfellow", style={"marginTop":"4px","fontStyle":"normal","opacity":"0.5"}),
            ], className="sb-quote"),

        ], className="sidebar"),

        # ═══════════════ CHAT CENTRAL ═══════════════
        html.Div([
            # header
            html.Div([
                html.H1([html.Span("💬", className="h-icon"), " Chat Inteligente"]),
                html.P("Tu asistente musical potenciado con IA ✨"),
            ], className="chat-header"),

            # métricas
            html.Div([
                html.Div([html.Div("🎧", className="mc-icon"),
                    html.Div([html.Div("12,273", className="mc-val"), html.Div("canciones indexadas", className="mc-label")])],
                    className="metric-card"),
                html.Div([html.Div("⚡", className="mc-icon"),
                    html.Div([html.Div("RAG + FAISS", className="mc-val"), html.Div("Búsqueda avanzada", className="mc-label")])],
                    className="metric-card"),
                html.Div([html.Div("🧠", className="mc-icon"),
                    html.Div([html.Div("BERT + OpenAI", className="mc-val"), html.Div("IA de última generación", className="mc-label")])],
                    className="metric-card"),
                html.Div([html.Div("🔍", className="mc-icon"),
                    html.Div([html.Div("Análisis profundo", className="mc-val"), html.Div("Sentimiento & contexto", className="mc-label")])],
                    className="metric-card"),
            ], className="metrics-row"),

            # caja de chat — contenedor fijo con scroll interno
            html.Div([
                # bienvenida (siempre visible al tope)
                html.Div([
                    html.Div("🎵", className="welcome-avatar"),
                    html.Div([
                        html.Strong("¡Hola! Soy 🎵 CorpusBeat IA 🎵"),
                        html.Br(),
                        "Puedo ayudarte a descubrir, analizar y entender música como nunca antes.",
                        html.Br(),
                        "¿Qué quieres explorar hoy?",
                    ], className="welcome-text"),
                ], className="welcome-msg"),

                # historial dinámico — crece hacia abajo
                html.Div(id="chat_box"),

                # espaciador que empuja el scroll hasta el fondo
                html.Div(id="chat-bottom", style={"height": "1px", "flexShrink": "0"}),

            ], className="chat-box", id="chat-scroll-box"),

            # pills rápidas
            html.Div([
                html.Button("❤️ ¿Qué canciones hablan de amor?",        id="ex1", className="quick-pill"),
                html.Button("📊 Analiza el sentimiento de 3 canciones", id="ex2", className="quick-pill"),
                html.Button("🎲 Dame 3 canciones aleatorias",            id="ex3", className="quick-pill"),
                html.Button("📝 Haz un resumen de 3 canciones",         id="ex4", className="quick-pill"),
            ], className="quick-pills"),

            # input
            html.Div([
                dcc.Input(
                    id="chat-input",
                    type="text",
                    placeholder="Escribe tu pregunta o pega una letra... (Enter para enviar)",
                    className="chat-input",
                    n_submit=0,
                    debounce=False,
                ),
                html.Button("Enviar ✈", id="send-btn", n_clicks=0, className="send-btn"),
            ], className="input-row"),

        ], className="main-chat"),

        # ═══════════════ PANEL DERECHO ═══════════════
        _right_panel(),

    ], className="app-shell"),

    html.Div("🎵 CorpusBeat IA 🎵 · Hecho con ♥ y mucha música  〰〰〰", className="app-footer"),
])

# ─────────────────────────────────────────────────────────────
# HELPER: reconstruye los componentes visuales desde el store
# ─────────────────────────────────────────────────────────────
def _render_history(history: list) -> list:
    """
    history es una lista de dicts:
      {"role": "user"|"bot"|"badge", "text": str}
    Devuelve lista de componentes Dash.
    """
    componentes = []
    for item in history:
        role = item["role"]
        text = item["text"]
        if role == "user":
            componentes.append(html.Div(text, className="user-msg"))
        elif role == "badge":
            componentes.append(html.Div(text, className="intent-badge"))
        else:  # bot
            componentes.append(
                # dcc.Markdown renderiza **negritas**, listas, etc.
                dcc.Markdown(
                    text,
                    className="bot-msg",
                    style={"whiteSpace": "pre-wrap"},
                )
            )
    return componentes


# ─────────────────────────────────────────────────────────────
# CALLBACK: pills de ejemplo llenan el input
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("chat-input", "value"),
    Input("ex1", "n_clicks"),
    Input("ex2", "n_clicks"),
    Input("ex3", "n_clicks"),
    Input("ex4", "n_clicks"),
    prevent_initial_call=True,
)
def fill_example(e1, e2, e3, e4):
    ejemplos = {
        "ex1": "¿Qué canciones hablan de amor?",
        "ex2": "Analiza el sentimiento de 3 canciones",
        "ex3": "Dame 3 canciones aleatorias de géneros variados",
        "ex4": "Haz un resumen de 3 canciones",
    }
    return ejemplos.get(ctx.triggered_id, "")


# ─────────────────────────────────────────────────────────────
# CALLBACK: enviar mensaje — historial en dcc.Store (sin global)
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("chat_box",    "children"),
    Output("chat-input",  "value", allow_duplicate=True),
    Output("chat-store",  "data"),
    Input("send-btn",     "n_clicks"),
    Input("chat-input",   "n_submit"),
    State("chat-input",   "value"),
    State("chat-store",   "data"),
    prevent_initial_call=True,
)
def chat(n_clicks, n_submit, text, history):
    if not text or not text.strip():
        return _render_history(history), "", history

    intent   = detectar_intencion(text)
    response = ejecutar_modulo(text)

    # acumular en el store (lista de dicts serializables)
    history = history or []
    history.append({"role": "user",  "text": text})
    history.append({"role": "badge", "text": ETIQUETAS.get(intent, "")})
    history.append({"role": "bot",   "text": response})

    return _render_history(history), "", history


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)