from pymongo import MongoClient
import pandas as pd


class MongoReader:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def connect(self):
        """Conectar a MongoDB (validando conexión real)"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)

            # 🔥 FORZAR conexión real
            self.client.server_info()

            db = self.client[self.db_name]
            self.collection = db[self.collection_name]

            print("✅ Conexión REAL establecida")

        except Exception as e:
            print(f"❌ Error conectando a MongoDB: {e}")
            self.client = None
            self.collection = None
            raise

    def fetch_all(self) -> pd.DataFrame:
        """Obtiene todos los documentos como DataFrame"""
        docs = list(self.collection.find())
        df = pd.DataFrame(docs)
        print(f"📥 Documentos cargados: {len(df)}")
        return df

    def run(self) -> pd.DataFrame:
        """Pipeline completo"""
        self.connect()
        return self.fetch_all()