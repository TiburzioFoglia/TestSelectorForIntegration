import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import List

class CodeBERTEmbedder:
    """Classe responsabile solo dell'estrazione degli embeddings"""

    def __init__(self, model_name="microsoft/codebert-base"):
        print(f"Caricamento del modello {model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Modello caricato su: {self.device}")

    def get_code_embeddings(self, code_snippets: List[str], max_length: int = 512) -> np.ndarray:
        """Estrae embeddings da una lista di code snippets"""
        embeddings = []
        print(f"Elaborazione di {len(code_snippets)} snippet di codice...")

        for i, code in enumerate(code_snippets):

            print(f"Processati {i}/{len(code_snippets)} snippet")

            # Tokenizza il codice
            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )

            # Sposta su GPU se disponibile
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Ottieni embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usa il token [CLS] per l'embedding del codice intero
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)