from .embedders.codebert_embedder import CodeBERTEmbedder
from .embedders.codeT5_embedder import CodeT5Embedder
from .embedders.polyCoder_embedder import PolyCoderEmbedder
from .embedders.sentenceTransformer_embedder import SentenceTransformerEmbedder
from .embedders.unixCoder_embedder import UnixCoderEmbedder
from .embedders.starCoder2_embedder import StarCoder2Embedder

def get_embedder(embedder_name: str):
    if embedder_name == "codeBert":
        return CodeBERTEmbedder("microsoft/codebert-base")
    if embedder_name == "graphCodeBert":
        return CodeBERTEmbedder("microsoft/graphcodebert-base")
    elif embedder_name == "codeT5":
        return CodeT5Embedder()
    elif embedder_name == "polyCoder":
        return PolyCoderEmbedder()
    elif embedder_name == "sentenceTransformer":
        return SentenceTransformerEmbedder()
    elif embedder_name == "unixCoder":
        return UnixCoderEmbedder()
    elif embedder_name == "starCoder2":
        return StarCoder2Embedder()
    else:
        raise ValueError(f"Embedder '{embedder_name}' non supportato.")

