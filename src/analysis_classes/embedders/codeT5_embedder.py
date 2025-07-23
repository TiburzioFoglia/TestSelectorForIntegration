import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List


class CodeT5Embedder:
    """Classe per estrarre embeddings con CodeT5."""

    def __init__(self, model_name="Salesforce/codet5p-110m-embedding"):
        print(f"Caricamento del modello {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Modello caricato su: {self.device}")

    def get_code_embeddings(self, code_snippets: List[str], max_length: int = 512) -> np.ndarray:
        """Estrae embeddings da una lista di code snippets."""
        embeddings = []
        print(f"Elaborazione di {len(code_snippets)} snippet di codice...")

        for i, code in enumerate(code_snippets):
            print(f"Processati {i}/{len(code_snippets)} snippet")

            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.cpu().numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)