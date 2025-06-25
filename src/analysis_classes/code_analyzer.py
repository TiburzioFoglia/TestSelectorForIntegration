import os

import numpy as np
from typing import Dict
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .cluster_analyzer import CodeClusterAnalyzer
from .complexity_analyzer import CodeComplexityAnalyzer
from .codebert_embedder import CodeBERTEmbedder


class ComprehensiveCodeAnalyzer:
    """Classe principale che coordina tutte le analisi"""

    def __init__(self):
        self.embedder = CodeBERTEmbedder()
        self.cluster_analyzer = CodeClusterAnalyzer()
        self.complexity_analyzer = CodeComplexityAnalyzer()

    def analyze_extracted_methods(self, methods_file: str, output_dir: str):
        """Analisi completa dei metodi estratti"""

        # Carica i metodi
        try:
            with open(methods_file, 'r', encoding='utf-8') as f:
                code_snippets = json.load(f)
            print(f"Caricati {len(code_snippets)} metodi per l'analisi")
        except FileNotFoundError:
            print(f"File {methods_file} non trovato. Esegui prima l'estrazione.")
            return

        # Estrai embeddings
        print("\nEstrazione embeddings ...")
        embeddings = self.embedder.get_code_embeddings(code_snippets)

        # Analisi complessità
        print("\nAnalisi complessita' ...")
        complexity_metrics = self.complexity_analyzer.analyze_code_complexity(code_snippets)

        # Crea DataFrame
        method_names = [f'Method_{i}' for i in range(len(code_snippets))]
        df = pd.DataFrame({
            'method_name': method_names,
            'lines_of_code': [m['lines_of_code'] for m in complexity_metrics],
            'cyclomatic_complexity': [m['cyclomatic_complexity'] for m in complexity_metrics],
            'nesting_depth': [m['nesting_depth'] for m in complexity_metrics],
            'method_calls': [m['method_calls'] for m in complexity_metrics],
            'cognitive_complexity': [m['cognitive_complexity'] for m in complexity_metrics],
        })

        # Analizza similarità con diverse metriche
        print("\nAnalisi similarita' ...")
        similarity_results = {}
        for metric in ['cosine', 'euclidean', 'manhattan']:
            similarity_results[metric] = self.cluster_analyzer.find_similar_methods(
                embeddings, method_names, metric=metric
            )

        # Clustering comprehensive
        print("\nClustering ...")
        clustering_results = self.cluster_analyzer.cluster_methods_comprehensive(embeddings)

        # Trova il miglior risultato
        best_alg, best_metric, best_labels = self.cluster_analyzer.get_best_clustering_result()
        print(f"\nMiglior clustering: {best_alg} con metrica {best_metric}")

        # Crea visualizzazioni
        print("\nCreazione grafici ...")
        self.cluster_analyzer.plot_clustering_results(embeddings, output_dir)
        self.cluster_analyzer.plot_hierarchical_dendrogram(embeddings, output_dir,'cosine')
        self.cluster_analyzer.plot_hierarchical_dendrogram(embeddings, output_dir,'euclidean')

        # Analizza caratteristiche dei cluster
        cluster_characteristics = self.cluster_analyzer.analyze_cluster_characteristics(df, best_labels)

        # Salva risultati
        results = {
            'method_analysis': df.to_dict('records'),
            'clustering_results': {
                alg: {metric: {'labels': result['labels'].tolist() if result else None,
                               'n_clusters': result['n_clusters'] if result else 0,
                               'n_outliers': result['n_outliers'] if result else 0,
                               'quality_score': result['quality_score'] if result else 0}
                      for metric, result in alg_results.items()}
                for alg, alg_results in clustering_results.items()
            },
            'best_clustering': {
                'algorithm': best_alg,
                'metric': best_metric,
                'labels': best_labels.tolist() if best_labels is not None else []
            },
            'similarity_analysis': {
                metric: {method: [(sim_method, float(score)) for sim_method, score in similarities[:3]]
                         for method, similarities in results.items()}
                for metric, results in similarity_results.items()
            },
            'cluster_characteristics': cluster_characteristics,
            'complexity_summary': {
                'avg_cyclomatic': df['cyclomatic_complexity'].mean(),
                'avg_lines': df['lines_of_code'].mean(),
                'avg_cognitive': df['cognitive_complexity'].mean(),
                'complexity_distribution': df['cyclomatic_complexity'].describe().to_dict()
            }
        }

        # Salva risultati
        output_path = os.path.join(output_dir, 'comprehensive_analysis_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print("\nCreazione grafici aggiuntivi ...")
        # Crea grafici di confronto
        self.create_comparison_plots(embeddings, results['clustering_results'], df, output_dir)

        # Crea grafici di analisi complessità
        best_labels = np.array(results['best_clustering']['labels'])
        self.create_complexity_analysis_plots(df, best_labels, output_dir)


        # Mostra summary dei risultati
        #self._print_analysis_summary(results, len(code_snippets))

        return results

    def _print_analysis_summary(self, results: Dict, n_methods: int):
        """Stampa un summary dei risultati dell'analisi"""

        print("\n" + "=" * 60)
        print("== SUMMARY ANALISI COMPREHENSIVE ==")
        print("=" * 60)

        print(f"📊 Metodi analizzati: {n_methods}")

        # Clustering summary
        best = results['best_clustering']
        print(f"🎯 Miglior clustering: {best['algorithm'].upper()} con {best['metric']}")

        # Conta cluster del miglior risultato
        best_labels = best['labels']
        if best_labels:
            unique_clusters = set(best_labels)
            n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            n_outliers = sum(1 for l in best_labels if l == -1)
            print(f"   → {n_clusters} cluster identificati")
            print(f"   → {n_outliers} metodi outlier ({n_outliers / n_methods * 100:.1f}%)")

        # Complessità summary
        complexity = results['complexity_summary']
        print(f"📈 Complessità media:")
        print(f"   → Ciclomatica: {complexity['avg_cyclomatic']:.1f}")
        print(f"   → Cognitiva: {complexity['avg_cognitive']:.1f}")
        print(f"   → Linee di codice: {complexity['avg_lines']:.1f}")

        # Clustering comparison
        print(f"\n🔍 Confronto algoritmi di clustering:")
        clustering_results = results['clustering_results']

        for algorithm in clustering_results:
            print(f"\n   {algorithm.upper()}:")
            for metric, result in clustering_results[algorithm].items():
                if result and result['n_clusters'] > 0:
                    print(f"     {metric}: {result['n_clusters']} cluster, "
                          f"qualità {result['quality_score']:.3f}")

        # Top similarità
        print(f"\n🔗 Esempi di metodi simili (metrica cosine):")
        similarity = results['similarity_analysis'].get('cosine', {})
        for i, (method, similarities) in enumerate(list(similarity.items())[:3]):
            if similarities:
                sim_method, score = similarities[0]
                print(f"   {method} ↔ {sim_method} (similarità: {score:.3f})")

        print(f"\n💾 Risultati salvati in: analysis_output/comprehensive_analysis_results.json")
        print("📊 Grafici salvati in: analysis_output/")
        print("=" * 60)

    def create_comparison_plots(self, embeddings: np.ndarray, clustering_results: Dict,
                                df: pd.DataFrame, output_dir: str):
        """Crea grafici di confronto tra diversi algoritmi e metriche"""

        # 1. Confronto qualità clustering
        plt.figure(figsize=(15, 5))

        # Subplot 1: Heatmap qualità
        plt.subplot(1, 3, 1)
        quality_matrix = []
        algorithms = []
        metrics = []

        for alg_name, alg_results in clustering_results.items():
            if any(alg_results.values()):
                algorithms.append(alg_name)
                quality_row = []
                for metric in ['cosine', 'euclidean', 'manhattan']:
                    result = alg_results.get(metric)
                    quality = result['quality_score'] if result else 0
                    quality_row.append(quality)
                quality_matrix.append(quality_row)

        if algorithms:
            metrics = ['cosine', 'euclidean', 'manhattan']
            sns.heatmap(quality_matrix, annot=True, fmt='.3f',
                        xticklabels=metrics, yticklabels=algorithms,
                        cmap='RdYlBu_r', center=0)
            plt.title('Qualità Clustering\n(Silhouette Score)')

        # Subplot 2: Numero di cluster
        plt.subplot(1, 3, 2)
        cluster_counts = []
        for alg_name in algorithms:
            alg_results = clustering_results[alg_name]
            cluster_row = []
            for metric in metrics:
                result = alg_results.get(metric)
                n_clusters = result['n_clusters'] if result else 0
                cluster_row.append(n_clusters)
            cluster_counts.append(cluster_row)

        if algorithms:
            sns.heatmap(cluster_counts, annot=True, fmt='d',
                        xticklabels=metrics, yticklabels=algorithms,
                        cmap='viridis')
            plt.title('Numero di Cluster')

        # Subplot 3: Percentuale outliers
        plt.subplot(1, 3, 3)
        outlier_percentages = []
        for alg_name in algorithms:
            alg_results = clustering_results[alg_name]
            outlier_row = []
            for metric in metrics:
                result = alg_results.get(metric)
                if result:
                    total_samples = result['n_clusters'] * 10  # Stima approssimativa
                    outlier_pct = (result['n_outliers'] / len(df)) * 100
                else:
                    outlier_pct = 0
                outlier_row.append(outlier_pct)
            outlier_percentages.append(outlier_row)

        if algorithms:
            sns.heatmap(outlier_percentages, annot=True, fmt='.1f',
                        xticklabels=metrics, yticklabels=algorithms,
                        cmap='Reds')
            plt.title('Percentuale Outliers (%)')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'clustering_comparison_heatmaps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

    def create_complexity_analysis_plots(self, df: pd.DataFrame, best_labels: np.ndarray, output_dir: str):
        """Crea grafici di analisi della complessità per cluster"""

        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = best_labels

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Distribuzione complessità per cluster
        regular_clusters = [c for c in set(best_labels) if c != -1]
        if regular_clusters:
            complexity_data = [df_with_clusters[df_with_clusters['cluster'] == c]['cyclomatic_complexity'].values
                               for c in regular_clusters]
            axes[0, 0].boxplot(complexity_data)
            axes[0, 0].set_title('Complessità Ciclomatica per Cluster')
            axes[0, 0].set_xlabel('Cluster')
            axes[0, 0].set_ylabel('Complessità Ciclomatica')
            axes[0, 0].set_xticklabels([f'C{c}' for c in regular_clusters])

        # 2. Scatter plot complessità vs linee di codice
        scatter = axes[0, 1].scatter(df_with_clusters['lines_of_code'],
                                     df_with_clusters['cyclomatic_complexity'],
                                     c=df_with_clusters['cluster'],
                                     cmap='tab10', alpha=0.7)
        axes[0, 1].set_title('Complessità vs Linee di Codice')
        axes[0, 1].set_xlabel('Linee di Codice')
        axes[0, 1].set_ylabel('Complessità Ciclomatica')
        plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')

        # 3. Complessità cognitiva per cluster
        if regular_clusters:
            cognitive_data = [df_with_clusters[df_with_clusters['cluster'] == c]['cognitive_complexity'].values
                              for c in regular_clusters]
            axes[0, 2].boxplot(cognitive_data)
            axes[0, 2].set_title('Complessità Cognitiva per Cluster')
            axes[0, 2].set_xlabel('Cluster')
            axes[0, 2].set_ylabel('Complessità Cognitiva')
            axes[0, 2].set_xticklabels([f'C{c}' for c in regular_clusters])

        # 4. Distribuzione profondità nesting
        axes[1, 0].hist(df_with_clusters['nesting_depth'], bins=15, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribuzione Profondità Nesting')
        axes[1, 0].set_xlabel('Profondità Nesting')
        axes[1, 0].set_ylabel('Frequenza')

        # 5. Chiamate di metodo per cluster
        if regular_clusters:
            calls_data = [df_with_clusters[df_with_clusters['cluster'] == c]['method_calls'].values
                          for c in regular_clusters]
            axes[1, 1].boxplot(calls_data)
            axes[1, 1].set_title('Chiamate di Metodo per Cluster')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Numero Chiamate')
            axes[1, 1].set_xticklabels([f'C{c}' for c in regular_clusters])

        # 6. Correlation matrix delle metriche
        correlation_cols = ['lines_of_code', 'cyclomatic_complexity', 'nesting_depth',
                            'method_calls', 'cognitive_complexity']
        corr_matrix = df_with_clusters[correlation_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Correlazione tra Metriche')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'complexity_analysis_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
