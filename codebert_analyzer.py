import torch
from numpy import floating
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import List, Dict, Tuple, Any
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


class CodeBERTAnalyzer:
    def __init__(self, model_name="microsoft/codebert-base"):
        print(f"Caricamento del modello {model_name}...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        print(f"Modello caricato su: {self.device}")

    def get_code_embeddings(self, code_snippets: List[str], max_length: int = 512) -> np.ndarray:
        embeddings = []

        print(f"Elaborazione di {len(code_snippets)} snippet di codice...")

        for i, code in enumerate(code_snippets):
            if i % 10 == 0:
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

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        return cosine_similarity(embeddings)

    def find_similar_methods(self, embeddings: np.ndarray, method_names: List[str],
                             top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:

        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        similar_methods = {}

        for i, method_name in enumerate(method_names):
            # Ottieni gli indici dei metodi più simili (escluso se stesso)
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][1:top_k + 1]  # Escludi se stesso

            similar_list = []
            for idx in top_indices:
                similar_list.append((method_names[idx], similarities[idx]))

            similar_methods[method_name] = similar_list

        return similar_methods

    def cluster_methods(self, embeddings: np.ndarray, method: str = 'dbscan', **kwargs) -> np.ndarray:
        if method == 'dbscan':
            return self._dbscan_clustering(embeddings, **kwargs)
        elif method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 3)
            return self._kmeans_clustering(embeddings, n_clusters)
        else:
            raise ValueError(f"Metodo non supportato: {method}")

    def _dbscan_clustering(self, embeddings: np.ndarray, eps: float = None, min_samples: int = 2) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        # Auto-determina eps se non specificato
        if eps is None:
            eps = self._estimate_optimal_eps(embeddings, min_samples)
            print(f"NO Epsilon passed, using: {eps:.4f}")

        # Usa distanza coseno per embeddings
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(embeddings)

        # Statistiche clustering
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"DBSCAN ha trovato {n_clusters} cluster e {n_noise} outlier")

        return cluster_labels

    def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

    def _estimate_optimal_eps(self, embeddings: np.ndarray, min_samples: int) -> floating[Any]:
        from sklearn.neighbors import NearestNeighbors

        # Calcola distanze ai k-esimi vicini più prossimi
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
        neighbors_fit = neighbors.fit(embeddings)
        distances, indices = neighbors_fit.kneighbors(embeddings)

        # Ordina le distanze al min_samples-esimo vicino
        distances = np.sort(distances[:, min_samples - 1], axis=0)

        # Trova il "gomito" nella curva delle distanze
        # Usa il 90° percentile come euristica
        optimal_eps = np.percentile(distances, 90)

        return optimal_eps

    def analyze_code_complexity(self, code_snippets: List[str]) -> List[Dict]:
        complexity_metrics = []

        for code in code_snippets:
            lines = code.split('\n')
            metrics = {
                'lines_of_code': len([line for line in lines if line.strip()]),
                'cyclomatic_complexity': self._estimate_cyclomatic_complexity(code),
                'nesting_depth': self._calculate_nesting_depth(code),
                'method_calls': len(self._extract_method_calls(code))
            }
            complexity_metrics.append(metrics)

        return complexity_metrics

    def _estimate_cyclomatic_complexity(self, code: str) -> int:
        complexity_keywords = ['if', 'else', 'while', 'for', 'foreach', 'switch', 'case', 'catch', '&&', '||']
        complexity = 1  # Base complexity

        for keyword in complexity_keywords:
            complexity += code.lower().count(keyword)

        return complexity

    def _calculate_nesting_depth(self, code: str) -> int:
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
        import re
        pattern = r'\b\w+\s*\('
        matches = re.findall(pattern, code)
        return [match.replace('(', '').strip() for match in matches]

    def generate_analysis_report(self, methods_info: List[Dict], embeddings: np.ndarray,):

        method_names = [method['name'] for method in methods_info]
        code_snippets = [method['body'] for method in methods_info]

        # Calcola similarità
        similar_methods = self.find_similar_methods(embeddings, method_names)

        # Raggruppa in cluster
        cluster_labels = self._robust_clustering(embeddings)

        # Analizza complessità
        complexity_metrics = self.analyze_code_complexity(code_snippets)

        # Crea DataFrame per l'analisi
        df = pd.DataFrame({
            'method_name': method_names,
            'cluster': cluster_labels,
            'lines_of_code': [m['lines_of_code'] for m in complexity_metrics],
            'cyclomatic_complexity': [m['cyclomatic_complexity'] for m in complexity_metrics],
            'nesting_depth': [m['nesting_depth'] for m in complexity_metrics],
            'method_calls': [m['method_calls'] for m in complexity_metrics]
        })

        # Genera visualizzazioni
        self._create_visualizations(df)

        # Salva risultati
        results = {
            'method_analysis': df.to_dict('records'),
            'similar_methods': similar_methods,
            'cluster_analysis': self._analyze_clusters(df)
        }

        with open('analysis_output/codebert_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"Report salvato in: analysis_output/codebert_analysis_results.json")
        return results


    def _robust_clustering(self, embeddings: np.ndarray, max_iterations: int = 6) -> np.ndarray:
        n_samples = len(embeddings)

        # Parametri DBSCAN da testare progressivamente
        dbscan_configs = [
            {'eps': 0.5, 'min_samples': max(2, n_samples // 10)},   # Configurazione standard
            {'eps': 0.4, 'min_samples': max(2, n_samples // 15)},   # Più permissivo
            {'eps': 0.3, 'min_samples': max(2, n_samples // 20)},   # Ancora più permissivo
            {'eps': 0.2, 'min_samples': 2},                         # Molto permissivo ma almeno 2 elementi
            {'eps': 0.6, 'min_samples': max(2, n_samples // 8)},    # Eps più alto
            {'eps': 0.1, 'min_samples': 2},                         # Ultima chance con eps molto basso
        ]

        best_result = None
        best_score = -1

        print(f"Inizio clustering robusto su {n_samples} campioni...")

        for iteration in range(min(max_iterations, len(dbscan_configs))):
            config = dbscan_configs[iteration]
            print(f"Iterazione {iteration + 1}: DBSCAN con eps={config['eps']}, min_samples={config['min_samples']}")

            try:
                cluster_labels = self.cluster_methods(embeddings, method='dbscan', **config)
                result_info = self._evaluate_clustering_result(cluster_labels, n_samples)

                print(f"  → {result_info['n_clusters']} cluster, {result_info['n_outliers']} outliers "
                      f"({result_info['outlier_percentage']:.1f}%), score: {result_info['score']:.3f}")

                # Salva il miglior risultato basato su uno score composito
                if result_info['score'] > best_score:
                    if result_info['n_outliers'] > 0 and result_info['n_clusters'] > 1:
                        best_result = cluster_labels
                        best_score = result_info['score']

                # Condizioni di stop: risultato soddisfacente
                if self._is_clustering_satisfactory(result_info):
                    print(f"Clustering soddisfacente trovato all'iterazione {iteration + 1}")
                    return cluster_labels

            except Exception as e:
                print(f"  ✗ Errore con configurazione {config}: {e}")
                continue

        # Se nessun DBSCAN ha dato risultati soddisfacenti, usa il migliore o fallback
        if best_result is not None:
            print(f"Uso il miglior risultato DBSCAN (score: {best_score:.3f})")
            return best_result
        else:
            print("Nessun DBSCAN valido, uso K-Means come fallback")
            return self._kmeans_fallback(embeddings, n_samples)


    def _evaluate_clustering_result(self, cluster_labels: np.ndarray, n_samples: int) -> Dict:
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_outliers = sum(1 for label in cluster_labels if label == -1)
        outlier_percentage = (n_outliers / n_samples) * 100

        # Score composito che bilancia numero di cluster e percentuale di outliers
        if n_clusters == 0:
            score = 0.0
        else:
            # Penalizza troppi pochi cluster e troppi outliers
            cluster_score = min(n_clusters / max(2, n_samples // 10), 1.0)  # Normalizzato
            outlier_penalty = max(0, (outlier_percentage - 20) / 100)  # Penalità se >20% outliers
            score = cluster_score - outlier_penalty

        return {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_percentage,
            'score': score
        }


    def _is_clustering_satisfactory(self, result_info: Dict) -> bool:
        """Determina se il clustering è soddisfacente"""
        return (
                result_info['n_clusters'] >= 2 and  # Almeno 2 cluster
                result_info['outlier_percentage'] <= 30 and  # Non troppi outliers
                result_info['score'] > 0.3  # Score decente
        )

    def _kmeans_fallback(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        # Determina il numero ottimale di cluster per K-Means
        if n_samples < 10:
            n_clusters = 2
        elif n_samples < 30:
            n_clusters = 3
        elif n_samples < 100:
            n_clusters = min(5, n_samples // 10)
        else:
            n_clusters = min(8, n_samples // 20)

        print(f"K-Means fallback con {n_clusters} cluster")
        return self.cluster_methods(embeddings, method='kmeans', n_clusters=n_clusters)

    def _create_visualizations(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Distribuzione dei cluster (gestisce anche outliers)
        cluster_counts = df['cluster'].value_counts().sort_index()
        bars = axes[0, 0].bar(range(len(cluster_counts)), cluster_counts.values)

        # Colora gli outlier diversamente
        if -1 in cluster_counts.index:
            outlier_pos = list(cluster_counts.index).index(-1)
            bars[outlier_pos].set_color('red')

        axes[0, 0].set_title('Distribuzione dei Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Numero di Metodi')

        # Etichette personalizzate
        labels = [f'Outliers' if x == -1 else f'Cluster {x}' for x in cluster_counts.index]
        axes[0, 0].set_xticks(range(len(cluster_counts)))
        axes[0, 0].set_xticklabels(labels, rotation=45)

        # Complessità vs Linee di codice
        scatter = axes[0, 1].scatter(df['lines_of_code'], df['cyclomatic_complexity'],
                                     c=df['cluster'], cmap='viridis')
        axes[0, 1].set_title('Complessità vs Linee di Codice')
        axes[0, 1].set_xlabel('Linee di Codice')
        axes[0, 1].set_ylabel('Complessità Ciclomatica')
        plt.colorbar(scatter, ax=axes[0, 1])

        # Heatmap delle metriche per cluster (escludi outliers se pochi)
        cluster_metrics = df[df['cluster'] != -1].groupby('cluster')[['lines_of_code', 'cyclomatic_complexity',
                                                                      'nesting_depth', 'method_calls']].mean()

        if len(cluster_metrics) > 0:
            sns.heatmap(cluster_metrics.T, annot=True, ax=axes[1, 0], cmap='YlOrRd')
            axes[1, 0].set_title('Metriche Medie per Cluster')
        else:
            axes[1, 0].text(0.5, 0.5, 'Nessun cluster\nregolare trovato',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Metriche per Cluster')

        # Distribuzione della complessità (escludi outliers per boxplot)
        regular_clusters = df[df['cluster'] != -1]['cluster'].unique()
        if len(regular_clusters) > 0:
            complexity_data = [df[(df['cluster'] == i) & (df['cluster'] != -1)]['cyclomatic_complexity'].values
                               for i in regular_clusters]
            axes[1, 1].boxplot(complexity_data)
            axes[1, 1].set_title('Distribuzione Complessità per Cluster')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Complessità Ciclomatica')
            axes[1, 1].set_xticklabels([f'C{i}' for i in regular_clusters])
        else:
            axes[1, 1].text(0.5, 0.5, 'Nessun cluster\nregolare per boxplot',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Distribuzione Complessità')

        plt.tight_layout()
        plt.savefig('analysis_output/codebert_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        # plt.show()

    def _analyze_clusters(self, df: pd.DataFrame) -> Dict:
        cluster_analysis = {}

        # Gestisci outlier (-1) separatamente in DBSCAN
        unique_clusters = set(df['cluster'].values)
        outliers = df[df['cluster'] == -1] if -1 in unique_clusters else pd.DataFrame()

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Outliers
                if len(outliers) > 0:
                    analysis = {
                        'method_count': len(outliers),
                        'avg_complexity': outliers['cyclomatic_complexity'].mean(),
                        'avg_lines': outliers['lines_of_code'].mean(),
                        'methods': outliers['method_name'].tolist(),
                        'characteristics': "Metodi outlier (unici/anomali)"
                    }
                    cluster_analysis['outliers'] = analysis
            else:
                cluster_data = df[df['cluster'] == cluster_id]

                analysis = {
                    'method_count': len(cluster_data),
                    'avg_complexity': cluster_data['cyclomatic_complexity'].mean(),
                    'avg_lines': cluster_data['lines_of_code'].mean(),
                    'methods': cluster_data['method_name'].tolist(),
                    'characteristics': self._describe_cluster(cluster_data)
                }

                cluster_analysis[f'cluster_{cluster_id}'] = analysis

        return cluster_analysis

    def _describe_cluster(self, cluster_data: pd.DataFrame) -> str:
        avg_complexity = cluster_data['cyclomatic_complexity'].mean()
        avg_lines = cluster_data['lines_of_code'].mean()

        if avg_complexity < 5 and avg_lines < 20:
            return "Metodi semplici e corti"
        elif avg_complexity >= 10 or avg_lines >= 50:
            return "Metodi complessi e lunghi"
        else:
            return "Metodi di complessità media"


# Funzione principale per integrare con l'estrattore precedente
def analyze_extracted_methods():
    # Carica i metodi estratti
    try:
        with open('methods_for_codebert.json', 'r', encoding='utf-8') as f:
            code_snippets = json.load(f)

        print(f"Caricati {len(code_snippets)} metodi per l'analisi")
    except FileNotFoundError:
        print("File methods_for_codebert.json non trovato. Esegui prima l'estrazione.")
        return

    # Inizializza CodeBERT
    analyzer = CodeBERTAnalyzer()

    # Ottieni embeddings
    embeddings = analyzer.get_code_embeddings(code_snippets)

    # Crea info sui metodi (assumendo nomi generici)
    methods_info = [
        {'name': f'TestMethod_{i}', 'body': code}
        for i, code in enumerate(code_snippets)
    ]

    # Genera report completo
    results = analyzer.generate_analysis_report(methods_info, embeddings)

    # 6. Mostra alcuni risultati interessanti
    print("\n=== RISULTATI ANALISI CODEBERT ===")
    print(f"Analizzati {len(code_snippets)} metodi di test")
    print(f"Identificati {len(set(results['method_analysis'][0]['cluster']))} cluster principali")

    # Mostra metodi più simili
    print("\n--- Metodi più simili ---")
    similar = results['similar_methods']
    for method, similarities in list(similar.items())[:3]:
        print(f"\n{method}:")
        for sim_method, score in similarities[:2]:
            print(f"  - {sim_method}: {score:.3f}")

    return results