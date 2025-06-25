import os

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import silhouette_score


class CodeClusterAnalyzer:
    """Classe responsabile del clustering e analisi dei pattern"""

    def __init__(self):
        self.clustering_results = {}
        self.metrics_available = ['cosine', 'euclidean', 'manhattan']
        self.algorithms_available = ['dbscan', 'kmeans', 'hierarchical']

    def cluster_methods_comprehensive(self, embeddings: np.ndarray,
                                      algorithms: List[str] = None,
                                      metrics: List[str] = None) -> Dict[str, Dict]:
        """Esegue clustering con diversi algoritmi e metriche"""

        if algorithms is None:
            algorithms = self.algorithms_available
        if metrics is None:
            metrics = self.metrics_available

        results = {}

        for algorithm in algorithms:
            results[algorithm] = {}

            for metric in metrics:
                print(f"Clustering con {algorithm} usando metrica {metric}...")

                try:
                    if algorithm == 'dbscan':
                        labels = self._dbscan_clustering(embeddings, metric)
                    elif algorithm == 'kmeans':
                        labels = self._kmeans_clustering(embeddings, metric)
                    elif algorithm == 'hierarchical':
                        labels = self._hierarchical_clustering(embeddings, metric)

                    # Valuta la qualità del clustering
                    quality_score = self._evaluate_clustering_quality(embeddings, labels, metric)

                    results[algorithm][metric] = {
                        'labels': labels,
                        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                        'n_outliers': sum(1 for l in labels if l == -1),
                        'quality_score': quality_score
                    }

                    print(f"  → {results[algorithm][metric]['n_clusters']} cluster, "
                          f"{results[algorithm][metric]['n_outliers']} outliers, "
                          f"qualità: {quality_score:.3f}")

                except Exception as e:
                    print(f"  ✗ Errore con {algorithm}-{metric}: {e}")
                    results[algorithm][metric] = None

        self.clustering_results = results
        return results

    def _dbscan_clustering(self, embeddings: np.ndarray, metric: str) -> np.ndarray:
        """DBSCAN clustering con ottimizzazione eps per ogni metrica"""
        optimal_eps = self._calculate_optimal_eps(embeddings, metric)

        # Parametri DBSCAN da testare progressivamente
        dbscan_configs = [
            {'eps': optimal_eps, 'min_samples': max(2, len(embeddings) // 10)},  # Configurazione ottimale
            {'eps': optimal_eps * 0.8, 'min_samples': max(2, len(embeddings) // 15)},  # Più restrittivo
            {'eps': optimal_eps * 1.2, 'min_samples': max(2, len(embeddings) // 20)},  # Più permissivo
            {'eps': optimal_eps * 0.6, 'min_samples': 2},  # Molto restrittivo
            {'eps': optimal_eps * 1.5, 'min_samples': max(2, len(embeddings) // 8)},  # Molto permissivo
            {'eps': optimal_eps * 0.4, 'min_samples': 2},  # Ultima chance restrittiva
        ]

        # Caso base (prima riga)
        dbscan = DBSCAN(eps=dbscan_configs[0]['eps'], min_samples=dbscan_configs[0]['min_samples'], metric=metric)
        base_case_cluster_labels = dbscan.fit_predict(embeddings)

        n_clusters = len(base_case_cluster_labels) - (1 if -1 in base_case_cluster_labels else 0)
        n_outliers = sum(1 for label in base_case_cluster_labels if label == -1)
        outlier_percentage = (n_outliers / len(embeddings)) * 100

        # Se ho almeno 2 clusters e meno del 30% di metodi outliers
        if n_clusters > 2 and outlier_percentage < 30:
            print(f"Eps ottimale calcolato: {optimal_eps:.4f}")
            return base_case_cluster_labels


        for iteration in range(1, len(dbscan_configs)):
            config = dbscan_configs[iteration]
            dbscan = DBSCAN(eps=config['eps'], min_samples=config['min_samples'], metric=metric)
            cluster_labels = dbscan.fit_predict(embeddings)

            n_clusters = len(cluster_labels) - (1 if -1 in cluster_labels else 0)
            n_outliers = sum(1 for label in cluster_labels if label == -1)
            outlier_percentage = (n_outliers / len(embeddings)) * 100

            if n_clusters > 2 and outlier_percentage < 30:
                print(f"Eps ottimale calcolato: {config['eps']:.4f}")
                return cluster_labels

        # Se tutto il resto ha fallito, uso il caso base
        print(f"Eps ottimale calcolato: {optimal_eps:.4f}")
        return base_case_cluster_labels

    def _kmeans_clustering(self, embeddings: np.ndarray, metric: str) -> np.ndarray:
        """K-Means clustering con numero ottimale di cluster"""
        n_clusters = self._determine_optimal_k(embeddings, metric)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)

    def _hierarchical_clustering(self, embeddings: np.ndarray, metric: str) -> np.ndarray:
        """Hierarchical clustering agglomerativo"""
        n_clusters = self._determine_optimal_k(embeddings, metric)

        if metric == 'cosine':
            linkage_method = 'average'
        elif metric == 'euclidean':
            linkage_method = 'ward'
        else:
            linkage_method = 'complete'

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=metric if metric != 'cosine' else 'euclidean'  # AgglomerativeClustering non supporta cosine con ward
        )
        return clustering.fit_predict(embeddings)

    def _calculate_optimal_eps(self, embeddings: np.ndarray, metric: str, k: int = None) -> float:
        """Calcola eps ottimale per DBSCAN usando k-distance plot"""
        n_samples = len(embeddings)

        if k is None:
            k = max(2, min(4, n_samples // 20))

        # Calcola le distanze k-nearest neighbor
        try:
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(embeddings)
            distances, indices = nbrs.kneighbors(embeddings)

            # Prendi la distanza al k-esimo vicino (ignora il punto stesso)
            k_distances = distances[:, k]
            k_distances = np.sort(k_distances)

            # Trova il punto di gomito
            optimal_eps = self._find_elbow_point(k_distances)

            # Aggiusta eps basandosi sulla metrica
            if metric == 'cosine':
                optimal_eps = min(optimal_eps, 0.1)  # Cosine è in [0,1]
            elif metric == 'manhattan':
                optimal_eps = min(optimal_eps * 1.2, 1.0)  # Manhattan tende ad avere distanze maggiori

            # Fallback se l'eps calcolato è troppo estremo
            median_dist = np.median(k_distances)
            if optimal_eps > median_dist * 3 or optimal_eps < median_dist * 0.1 :
                print(f"Eps calcolato ({optimal_eps:.4f}) sembra estremo, uso mediana: {median_dist:.4f}")
                optimal_eps = median_dist

            return optimal_eps

        except Exception as e:
            print(f"Errore nel calcolo eps per {metric}: {e}")
            # Fallback values basati sulla metrica
            fallback_values = {'cosine': 0.1, 'euclidean': 1.0, 'manhattan': 2.0}
            return fallback_values.get(metric, 1.0)

    def _determine_optimal_k(self, embeddings: np.ndarray, metric: str) -> int:
        """Determina numero ottimale di cluster usando elbow method"""
        n_samples = len(embeddings)
        max_k = min(10, n_samples // 3)

        if max_k < 2:
            return 2

        # Calcola inerzia per diversi k
        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Trova il gomito
        if len(inertias) > 2:
            # Calcola la derivata seconda per trovare il punto di massima curvatura
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            if len(diffs2) > 0:
                elbow_idx = np.argmax(diffs2) + 2  # +2 perché abbiamo fatto due diff
                optimal_k = min(k_range[elbow_idx], max_k)
            else:
                optimal_k = 3
        else:
            optimal_k = 3

        return optimal_k

    def _find_elbow_point(self, distances: np.ndarray) -> int:
        """Trova l'indice del punto di gomito in una curva di distanze"""
        n_points = len(distances)
        if n_points < 10:
            return min(n_points - 1, max(0, int(n_points * 0.75)))

        # Usa il metodo della linea retta per trovare il gomito
        # Traccia una linea dal primo all'ultimo punto
        x_coords = np.arange(n_points)
        y_coords = distances

        # Calcola la distanza di ogni punto dalla linea retta
        # Linea dal primo all'ultimo punto: y = mx + c
        x1, y1 = 0, y_coords[0]
        x2, y2 = n_points - 1, y_coords[-1]

        # Calcola la distanza perpendicolare di ogni punto dalla linea
        distances_from_line = []
        for i in range(n_points):
            x0, y0 = i, y_coords[i]
            # Distanza punto-linea: |ax + by + c| / sqrt(a² + b²)
            # Dove la linea è: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1

            distance = abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)
            distances_from_line.append(distance)

        # Il punto di gomito è quello con la massima distanza dalla linea
        # Ma considera solo la parte iniziale-media della curva
        search_end = min(n_points, int(n_points * 0.8))
        elbow_idx = np.argmax(distances_from_line[:search_end])

        # Assicura che non sia troppo all'inizio
        elbow_idx = max(elbow_idx, n_points // 20)

        return elbow_idx

    def _evaluate_clustering_quality(self, embeddings: np.ndarray, labels: np.ndarray, metric: str) -> float:
        """Valuta la qualità del clustering usando silhouette score"""

        try:
            unique_labels = set(labels)
            if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
                return 0.0

            # Rimuovi outliers per il calcolo del silhouette score
            mask = labels != -1
            if np.sum(mask) < 2:
                return 0.0

            score = silhouette_score(embeddings[mask], labels[mask], metric=metric)
            return score
        except:
            return 0.0

    def plot_clustering_results(self, embeddings: np.ndarray, output_dir: str):
        """Crea visualizzazioni comprehensive dei risultati di clustering"""

        if not self.clustering_results:
            print("Esegui prima il clustering!")
            return

        # Crea una griglia di subplot
        n_algorithms = len([alg for alg in self.clustering_results if any(self.clustering_results[alg].values())])
        n_metrics = len(self.metrics_available)

        fig = plt.figure(figsize=(20, 5 * n_algorithms))

        plot_idx = 1

        for alg_name, alg_results in self.clustering_results.items():
            if not any(alg_results.values()):
                continue

            for metric_name in self.metrics_available:
                if alg_results.get(metric_name) is None:
                    continue

                result = alg_results[metric_name]
                labels = result['labels']

                # Subplot per questo algoritmo/metrica
                ax = plt.subplot(n_algorithms, n_metrics, plot_idx)

                # Riduci dimensionalità per visualizzazione (PCA)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(embeddings)

                # Plot scatter con colori per cluster
                unique_labels = set(labels)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        # Outliers in nero
                        col = 'black'
                        marker = 'x'
                        label = 'Outliers'
                    else:
                        marker = 'o'
                        label = f'Cluster {k}'

                    class_member_mask = (labels == k)
                    xy = embeddings_2d[class_member_mask]
                    ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                               s=50, alpha=0.7, label=label)

                ax.set_title(f'{alg_name.upper()} - {metric_name}\n'
                             f'Clusters: {result["n_clusters"]}, '
                             f'Outliers: {result["n_outliers"]}, '
                             f'Quality: {result["quality_score"]:.3f}')
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)

                plot_idx += 1

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'clustering_comprehensive_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

    def plot_hierarchical_dendrogram(self, embeddings: np.ndarray, output_dir: str, metric: str = 'cosine'):
        """Crea dendrogram per clustering gerarchico"""

        plt.figure(figsize=(15, 8))

        # Calcola linkage matrix
        if metric == 'cosine':
            distances = pdist(embeddings, metric='cosine')
            linkage_matrix = linkage(distances, method='average')
        else:
            linkage_matrix = linkage(embeddings, method='ward', metric=metric)

        # Crea dendrogram
        dendrogram(linkage_matrix,
                   truncate_mode='level',
                   p=5,  # Mostra solo i primi 5 livelli
                   leaf_rotation=90,
                   leaf_font_size=10)

        plt.title(f'Dendrogram - Hierarchical Clustering ({metric} distance)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'dendrogram_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

    def get_best_clustering_result(self) -> Tuple[str, str, np.ndarray]:
        """Restituisce il miglior risultato di clustering basato sulla qualità"""

        best_score = -1
        best_algorithm = None
        best_metric = None
        best_labels = None

        for alg_name, alg_results in self.clustering_results.items():
            for metric_name, result in alg_results.items():
                if result is None:
                    continue

                score = result['quality_score']
                if score > best_score:
                    best_score = score
                    best_algorithm = alg_name
                    best_metric = metric_name
                    best_labels = result['labels']

        return best_algorithm, best_metric, best_labels

    def analyze_cluster_characteristics(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Analizza le caratteristiche di ogni cluster"""

        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels

        cluster_analysis = {}
        unique_clusters = set(labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == -1]
                cluster_analysis['outliers'] = {
                    'method_count': len(cluster_data),
                    'avg_complexity': cluster_data['cyclomatic_complexity'].mean() if len(cluster_data) > 0 else 0,
                    'avg_lines': cluster_data['lines_of_code'].mean() if len(cluster_data) > 0 else 0,
                    'methods': cluster_data['method_name'].tolist() if 'method_name' in cluster_data else [],
                    'characteristics': "Metodi outlier (unici/anomali)"
                }
            else:
                cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]

                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'method_count': len(cluster_data),
                    'avg_complexity': cluster_data['cyclomatic_complexity'].mean(),
                    'avg_lines': cluster_data['lines_of_code'].mean(),
                    'methods': cluster_data['method_name'].tolist() if 'method_name' in cluster_data else [],
                    'characteristics': self._describe_cluster_characteristics(cluster_data)
                }

        return cluster_analysis

    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame) -> str:
        """Descrivi le caratteristiche di un cluster"""
        avg_complexity = cluster_data['cyclomatic_complexity'].mean()
        avg_lines = cluster_data['lines_of_code'].mean()

        if avg_complexity < 5 and avg_lines < 20:
            return "Metodi semplici e corti"
        elif avg_complexity >= 10 or avg_lines >= 50:
            return "Metodi complessi e lunghi"
        else:
            return "Metodi di complessità media"

    def calculate_similarity_matrix(self, embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        """Calcola matrice di similarità con diverse metriche"""
        if metric == 'cosine':
            return cosine_similarity(embeddings)
        elif metric == 'euclidean':
            # Converte distanze in similarità (1 / (1 + distanza))
            distances = euclidean_distances(embeddings)
            return 1 / (1 + distances)
        elif metric == 'manhattan':
            distances = manhattan_distances(embeddings)
            return 1 / (1 + distances)
        else:
            raise ValueError(f"Metrica non supportata: {metric}")


    def find_similar_methods(self, embeddings: np.ndarray, method_names: List[str],
                             top_k: int = 5, metric: str = 'cosine') -> Dict[str, List[Tuple[str, float]]]:
        """Trova metodi simili usando diverse metriche"""
        similarity_matrix = self.calculate_similarity_matrix(embeddings, metric)
        similar_methods = {}

        for i, method_name in enumerate(method_names):
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][1:top_k + 1]  # Escludi se stesso

            similar_list = []
            for idx in top_indices:
                similar_list.append((method_names[idx], similarities[idx]))

            similar_methods[method_name] = similar_list

        return similar_methods