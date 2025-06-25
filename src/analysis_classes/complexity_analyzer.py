import torch
from numpy import floating
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import List, Dict, Tuple, Any
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings


class CodeComplexityAnalyzer:
    """Classe per l'analisi della complessità del codice"""

    def __init__(self):
        pass

    def analyze_code_complexity(self, code_snippets: List[str]) -> List[Dict]:
        """Analizza diverse metriche di complessità"""
        complexity_metrics = []

        for code in code_snippets:
            lines = code.split('\n')
            metrics = {
                'lines_of_code': len([line for line in lines if line.strip()]),
                'cyclomatic_complexity': self._estimate_cyclomatic_complexity(code),
                'nesting_depth': self._calculate_nesting_depth(code),
                'method_calls': len(self._extract_method_calls(code)),
                'cognitive_complexity': self._estimate_cognitive_complexity(code),
                'halstead_complexity': self._calculate_halstead_metrics(code)
            }
            complexity_metrics.append(metrics)

        return complexity_metrics

    def _estimate_cyclomatic_complexity(self, code: str) -> int:
        """Stima la complessità ciclomatica"""
        complexity_keywords = ['if', 'else', 'while', 'for', 'foreach', 'switch', 'case', 'catch', '&&', '||']
        complexity = 1  # Base complexity

        for keyword in complexity_keywords:
            complexity += code.lower().count(keyword)

        return complexity

    def _calculate_nesting_depth(self, code: str) -> int:
        """Calcola la profondità di nesting"""
        max_depth = 0
        current_depth = 0

        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1

        return max_depth

    def _extract_method_calls(self, code: str) -> List[str]:
        """Estrae le chiamate ai metodi"""
        import re
        pattern = r'\b\w+\s*\('
        matches = re.findall(pattern, code)
        return [match.replace('(', '').strip() for match in matches]

    def _estimate_cognitive_complexity(self, code: str) -> int:
        """Stima la complessità cognitiva (semplificata)"""
        # Implementazione semplificata della complessità cognitiva
        cognitive_keywords = {
            'if': 1, 'else': 1, 'while': 1, 'for': 1, 'foreach': 1,
            'switch': 1, 'case': 1, 'catch': 2, 'finally': 1,
            '&&': 1, '||': 1, '?': 1  # operatori ternari e logici
        }

        complexity = 0
        nesting_level = 0

        lines = code.split('\n')
        for line in lines:
            line_lower = line.strip().lower()

            # Conta i livelli di nesting
            if '{' in line:
                nesting_level += line.count('{')
            if '}' in line:
                nesting_level -= line.count('}')
                nesting_level = max(0, nesting_level)

            # Aggiungi complessità per keywords
            for keyword, weight in cognitive_keywords.items():
                if keyword in line_lower:
                    # Incrementa complessità basata sul nesting
                    complexity += weight * (1 + nesting_level)

        return complexity

    def _calculate_halstead_metrics(self, code: str) -> Dict:
        """Calcola metriche di Halstead (versione semplificata)"""
        import re

        # Operatori comuni
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=',
                     '&&', '||', '!', '&', '|', '^', '<<', '>>', '++', '--']

        # Conta operatori unici e totali
        unique_operators = set()
        total_operators = 0

        for op in operators:
            count = code.count(op)
            if count > 0:
                unique_operators.add(op)
                total_operators += count

        # Operandi (semplificato: parole che non sono keywords)
        keywords = {'if', 'else', 'while', 'for', 'return', 'int', 'string', 'bool',
                    'public', 'private', 'void', 'class', 'namespace'}

        words = re.findall(r'\b\w+\b', code)
        operands = [w for w in words if w.lower() not in keywords]
        unique_operands = set(operands)

        n1 = len(unique_operators)  # numero di operatori unici
        n2 = len(unique_operands)  # numero di operandi unici
        N1 = total_operators  # numero totale di operatori
        N2 = len(operands)  # numero totale di operandi

        # Calcola metriche di Halstead
        vocabulary = n1 + n2
        length = N1 + N2

        if n2 > 0:
            estimated_length = n1 * np.log2(n1) + n2 * np.log2(n2) if n1 > 0 and n2 > 0 else 0
            volume = length * np.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume
        else:
            estimated_length = volume = difficulty = effort = 0

        return {
            'vocabulary': vocabulary,
            'length': length,
            'estimated_length': estimated_length,
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }