import json
import os


class SaveResults:

    def save_complete_results(self, results: dict, methods_info: list, output_dir: str):
        # Risultati principali
        with open(os.path.join(output_dir, "complete_analysis.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_methods': len(methods_info),
                    'clusters_found': len(set(r['cluster'] for r in results['method_analysis'])),
                    'avg_complexity': sum(r['cyclomatic_complexity'] for r in results['method_analysis']) / len(
                        results['method_analysis'])
                },
                'detailed_results': results,
                'methods_info': methods_info
            }, f, indent=2, ensure_ascii=False, default=str)

        # Report testuale
        self._generate_text_report(results, methods_info, output_dir)

        # Metodi per cluster
        self._save_methods_by_cluster(results, methods_info, output_dir)

    def _generate_text_report(self, results: dict, methods_info: list, output_dir: str):

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("REPORT ANALISI CODEBERT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Sommario generale
        report_lines.append("SOMMARIO GENERALE")
        report_lines.append("-" * 30)
        report_lines.append(f"Totale metodi analizzati: {len(methods_info)}")

        clusters = set(r['cluster'] for r in results['method_analysis'])
        report_lines.append(f"Cluster identificati: {len(clusters)}")

        avg_complexity = sum(r['cyclomatic_complexity'] for r in results['method_analysis']) / len(
            results['method_analysis'])
        report_lines.append(f"Complessità media: {avg_complexity:.2f}")
        report_lines.append("")

        # Analisi per cluster
        report_lines.append("ANALISI PER CLUSTER")
        report_lines.append("-" * 30)

        for cluster_name, cluster_info in results['cluster_analysis'].items():
            report_lines.append(f"\n{cluster_name.upper()}:")
            report_lines.append(f"  • Metodi: {cluster_info['method_count']}")
            report_lines.append(f"  • Complessità media: {cluster_info['avg_complexity']:.2f}")
            report_lines.append(f"  • Linee medie: {cluster_info['avg_lines']:.1f}")
            report_lines.append(f"  • Caratteristiche: {cluster_info['characteristics']}")
            report_lines.append(f"  • Lista metodi: {', '.join(cluster_info['methods'])}")

        # Metodi simili
        report_lines.append(f"\nMETODI PIÙ SIMILI")
        report_lines.append("-" * 30)

        for method, similarities in list(results['similar_methods'].items())[:5]:
            report_lines.append(f"\n{method}:")
            for sim_method, score in similarities[:3]:
                report_lines.append(f"  → {sim_method} (similarità: {score:.3f})")

        # Salva report
        with open(os.path.join(output_dir, "analysis_report.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

    def _save_methods_by_cluster(self, results: dict, methods_info: list, output_dir: str):

        cluster_dir = os.path.join(output_dir, "methods_by_cluster")
        os.makedirs(cluster_dir, exist_ok=True)

        # Raggruppa metodi per cluster
        methods_by_cluster = {}
        for method_analysis in results['method_analysis']:
            cluster = method_analysis['cluster']
            method_name = method_analysis['method_name']

            if cluster not in methods_by_cluster:
                methods_by_cluster[cluster] = []

            # Trova il metodo corrispondente
            method_info = next(m for m in methods_info if m['name'] == method_name)
            methods_by_cluster[cluster].append(method_info)

        # Salva ogni cluster
        for cluster_id, methods in methods_by_cluster.items():
            cluster_file = os.path.join(cluster_dir, f"cluster_{cluster_id}.json")
            with open(cluster_file, 'w', encoding='utf-8') as f:
                json.dump(methods, f, indent=2, ensure_ascii=False)

            # Salva anche i corpi dei metodi in file separati
            cluster_code_dir = os.path.join(cluster_dir, f"cluster_{cluster_id}_code")
            os.makedirs(cluster_code_dir, exist_ok=True)

            for method in methods:
                method_file = os.path.join(cluster_code_dir, f"{method['name']}.cs")
                with open(method_file, 'w', encoding='utf-8') as f:
                    f.write(method['body'])



