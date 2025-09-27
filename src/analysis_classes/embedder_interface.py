import abc
from typing import List
import numpy as np


class Embedder(abc.ABC):
    """
    An abstract base class for all embedder models.
    It defines the contract that all concrete embedders must follow.
    """

    @abc.abstractmethod
    def get_code_embeddings(self, code_snippets: List[str]) -> np.ndarray:
        """
        Takes a list of code snippets and returns a list of their vector embeddings.

        Args:
            code_snippets: A list of strings, where each string is a piece of code.

        Returns:
            A list of lists of floats, representing the embedding for each snippet.
        """
        pass