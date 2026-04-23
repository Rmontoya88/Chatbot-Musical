import os
import numpy as np
import pandas as pd
import pickle
import faiss

import plotly.express as px
import plotly.io as pio

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


class EmbeddingsEDA:
    def __init__(self, base_path, output_path="graficos", show=True, save=True):
        self.base_path = base_path
        self.output_path = output_path
        self.show = show
        self.save = save

        os.makedirs(self.output_path, exist_ok=True)

        pio.renderers.default = "notebook_connected"

        self.chunks = None
        self.embeddings = None
        self.index = None
        self.labels = None

    # =========================
    # CARGA DE DATOS
    # =========================
    def load_data(self):
        with open(f"{self.base_path}\\chunks_parrafos.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.embeddings = np.load(f"{self.base_path}\\embeddings_parrafos.npy")
        self.index = faiss.read_index(f"{self.base_path}\\indice_parrafos.faiss")

        print("✅ Datos cargados")
        print("Chunks:", len(self.chunks))
        print("Embeddings:", self.embeddings.shape)
        print("FAISS:", self.index.ntotal)

    # =========================
    # SAVE + SHOW
    # =========================
    def _handle_fig(self, fig, name):
        if self.show:
            fig.show()
        if self.save:
            fig.write_image(f"{self.output_path}/{name}.png")

    # =========================
    # 1. NORMA EMBEDDINGS
    # =========================
    def plot_norms(self):
        norms = np.linalg.norm(self.embeddings, axis=1)

        fig = px.histogram(norms, nbins=50,
                           title="Distribución de normas",
                           template="plotly_dark")
        self._handle_fig(fig, "normas_embeddings")

    # =========================
    # 2. DIMENSIONES
    # =========================
    def plot_dimensions(self):
        df = pd.DataFrame(self.embeddings[:, :20])

        fig = px.box(df,
                     title="Distribución primeras 20 dimensiones",
                     template="plotly_dark")
        self._handle_fig(fig, "distribucion_dimensiones")

    # =========================
    # 3. PCA CORRELACIÓN
    # =========================
    def plot_pca_corr(self):
        pca = PCA(n_components=10)
        data = pca.fit_transform(self.embeddings)

        df = pd.DataFrame(data)

        fig = px.imshow(df.corr(),
                        text_auto=True,
                        color_continuous_scale="RdBu",
                        title="Correlación PCA",
                        template="plotly_dark")

        self._handle_fig(fig, "pca_correlacion")

    # =========================
    # 4. DENSIDAD SEMÁNTICA
    # =========================
    def plot_density(self):
        nbrs = NearestNeighbors(n_neighbors=5).fit(self.embeddings)
        distances, _ = nbrs.kneighbors(self.embeddings)

        density = distances.mean(axis=1)

        fig = px.histogram(density, nbins=50,
                           title="Densidad semántica",
                           template="plotly_dark")

        self._handle_fig(fig, "densidad_semantica")

    # =========================
    # 5. MÉTODO DEL CODO
    # =========================
    def plot_elbow(self, max_k=15):
        inertias = []

        for k in range(1, max_k):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(self.embeddings)
            inertias.append(km.inertia_)

        fig = px.line(x=list(range(1, max_k)), y=inertias,
                      markers=True,
                      title="Método del codo",
                      template="plotly_dark")

        self._handle_fig(fig, "metodo_codo")

    # =========================
    # 6. CLUSTERING
    # =========================
    def clustering(self, k=4):
        emb = normalize(self.embeddings)

        km = KMeans(n_clusters=k, random_state=42)
        self.labels = km.fit_predict(emb)

        print("Clusters:", np.unique(self.labels))

    # =========================
    # 7. DENDROGRAMA
    # =========================
    def plot_dendrogram(self):
        sample = self.embeddings[:500]
        Z = linkage(sample, method="ward")

        plt.figure(figsize=(10, 5))
        dendrogram(Z, no_labels=True)
        plt.title("Dendrograma")

        if self.show:
            plt.show()

        if self.save:
            plt.savefig(f"{self.output_path}/dendrograma.png")

        plt.close()

    # =========================
    # 8. MAPA SEMÁNTICO
    # =========================
    def plot_semantic_map(self, sample_size=2000):
        emb = normalize(self.embeddings)

        idx = np.random.choice(len(emb), sample_size, replace=False)

        sample_emb = emb[idx]
        sample_labels = self.labels[idx]

        titles = [self.chunks[i][0] for i in idx]
        artists = [self.chunks[i][1] for i in idx]
        lyrics = [self.chunks[i][2] for i in idx]

        pca = PCA(n_components=2)
        emb2d = pca.fit_transform(sample_emb)

        self.df_sample = pd.DataFrame({
            "x": emb2d[:, 0],
            "y": emb2d[:, 1],
            "cluster": sample_labels.astype(str),
            "titulo": titles,
            "artista": artists,
            "letra": lyrics
        })

        self.df_sample["letra_corta"] = self.df_sample["letra"].str[:120]

        fig = px.scatter(self.df_sample,
                         x="x", y="y",
                         color="cluster",
                         hover_data=["titulo", "artista", "letra_corta"],
                         title="Mapa semántico",
                         template="plotly_dark")

        self._handle_fig(fig, "mapa_semantico")

    # =========================
    # 9. ARTISTAS POR CLUSTER
    # =========================
    def artistas_por_cluster(self):
        df_group = self.df_sample.groupby(["cluster", "artista"]) \
            .size().reset_index(name="count")

        print("\n🎤 Artistas dominantes por cluster")

        for c in sorted(self.df_sample["cluster"].unique()):
            print(f"\nCluster {c}")
            print(
                df_group[df_group["cluster"] == c]
                .sort_values("count", ascending=False)
                .head(5)
            )

    # =========================
    # 10. METADATA ANALISIS
    # =========================
    def metadata_analysis(self):
        df_meta = pd.DataFrame({
            "titulo": [x[0] for x in self.chunks],
            "artista": [x[1] for x in self.chunks],
        })

        # Top artistas
        fig = px.bar(
            df_meta["artista"].value_counts().head(20),
            title="Top artistas",
            template="plotly_dark"
        )
        self._handle_fig(fig, "top_artistas")

        # Longitud letras
        df_meta["longitud"] = [len(x[2].split()) for x in self.chunks]

        fig = px.histogram(
            df_meta,
            x="longitud",
            nbins=50,
            title="Longitud de letras",
            template="plotly_dark"
        )
        self._handle_fig(fig, "longitud_letras")

    # =========================
    # PIPELINE COMPLETO
    # =========================
    def run_all(self):
        self.load_data()
        self.plot_norms()
        self.plot_dimensions()
        self.plot_pca_corr()
        self.plot_density()
        self.plot_elbow()
        self.clustering()
        self.plot_dendrogram()
        self.plot_semantic_map()
        self.artistas_por_cluster()
        self.metadata_analysis()

        print("\n✅ EDA COMPLETO")
        print(f"📁 Gráficos en: {self.output_path}")