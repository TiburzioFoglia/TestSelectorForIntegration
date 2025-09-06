import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List

class StarCoder2Embedder:
    """Classe per estrarre embeddings di codice utilizzando il modello StarCoder2"""

    def __init__(self, model_name="bigcode/starcoder2-3b"):
        print(f"Caricamento del modello {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # serve a farlo pesare meno senza perdere troppa precisione
        dtype = torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval() # Imposta il modello in modalità di valutazione
        print(f"Modello caricato su: {self.device} con dtype: {dtype}")

    def get_code_embeddings(self, code_snippets: List[str], max_length: int = 1024) -> np.ndarray:
        """Estrae embeddings da una lista di snippet di codice"""
        embeddings = []
        print(f"Elaborazione di {len(code_snippets)} snippet di codice...")

        for i, code in enumerate(code_snippets):
            print(f"Processando snippet {i+1}/{len(code_snippets)}")

            # Tokenizza il codice
            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

                last_hidden_state = outputs.hidden_states[-1]

                embedding = torch.mean(last_hidden_state, dim=1).cpu().numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)
