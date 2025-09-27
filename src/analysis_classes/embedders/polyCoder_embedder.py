import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List
from src.analysis_classes.embedder_interface import Embedder

class PolyCoderEmbedder(Embedder):
    """Classe per estrarre embeddings con PolyCoder"""

    def __init__(self, model_name="NinedayWang/PolyCoder-2.7B"):
        print(f"Caricamento del modello {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Se non c'è un token di padding, si può usare il token di fine sequenza
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
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
                outputs = self.model(**inputs, output_hidden_states=True)
                # Estrai l'ultimo stato nascosto e fai una media dei token
                last_hidden_state = outputs.hidden_states[-1]
                embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)