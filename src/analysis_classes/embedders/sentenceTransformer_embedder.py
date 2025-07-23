from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class SentenceTransformerEmbedder:
    """Classe per estrarre embeddings con Sentence-Transformers generico"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Caricamento del modello {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Modello caricato.")

    def get_code_embeddings(self, code_snippets: List[str]) -> np.ndarray:
        """Estrae embeddings da una lista di code snippets."""
        print(f"Elaborazione di {len(code_snippets)} snippet di codice...")
        embeddings = self.model.encode(code_snippets, show_progress_bar=True)
        return embeddings