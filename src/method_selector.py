import json
from typing import Dict, Any


def select_round_robin_from_clusters(input_data: Dict[str, Any], n: int) -> Dict[str, Any]:
    # Estrai la sezione cluster_analysis
    cluster_analysis = input_data.get("cluster_characteristics", {})

    if not cluster_analysis:
        raise ValueError("Nessun cluster_analysis trovato nei dati")

    # Crea liste di metodi per ogni categoria (cluster + outliers)
    all_categories = {}

    for key, value in cluster_analysis.items():
        if "methods" in value:
            all_categories[key] = list(value["methods"])

    if not all_categories:
        raise ValueError("Nessun metodo trovato nei cluster o outliers")

    # Separa cluster da outliers
    clusters = {k: v for k, v in all_categories.items() if k.startswith("cluster_")}
    outliers = {k: v for k, v in all_categories.items() if k == "outliers"}

    selected_methods = {}
    cluster_indices = {name: 0 for name in clusters.keys()}

    # Inizializza le liste per ogni categoria
    for category in all_categories.keys():
        selected_methods[category] = []

    selected_count = 0

    # Prendi un elemento da ogni cluster
    for cluster_name in clusters.keys():
        if selected_count < n and cluster_indices[cluster_name] < len(clusters[cluster_name]):
            method = clusters[cluster_name][cluster_indices[cluster_name]]
            selected_methods[cluster_name].append(method)
            cluster_indices[cluster_name] += 1
            selected_count += 1

    # Prendi tutti gli outliers
    if "outliers" in outliers and selected_count < n:
        outliers_methods = outliers["outliers"]
        outliers_to_take = min(len(outliers_methods), n - selected_count)

        for i in range(outliers_to_take):
            method = outliers_methods[i]
            selected_methods["outliers"].append(method)
            selected_count += 1

    # Round-robin tra i cluster rimanenti
    if selected_count < n and clusters:
        cluster_names = list(clusters.keys())
        round_index = 0

        while selected_count < n:
            # Determina il cluster corrente
            current_cluster = cluster_names[round_index % len(cluster_names)]

            # Verifica se il cluster ha ancora elementi
            if cluster_indices[current_cluster] < len(clusters[current_cluster]):
                method = clusters[current_cluster][cluster_indices[current_cluster]]
                selected_methods[current_cluster].append(method)
                cluster_indices[current_cluster] += 1
                selected_count += 1

            round_index += 1

            # Se tutti i cluster sono esauriti, esci
            if all(cluster_indices[cluster] >= len(clusters[cluster]) for cluster in cluster_names):
                break

    # Costruisci il risultato finale
    result = {}

    for category_name in selected_methods:
        if selected_methods[category_name]:  # Solo se ci sono metodi selezionati
            result[category_name] = {
                "methods": selected_methods[category_name],
                "method_count": len(selected_methods[category_name])
            }

    return result


def process_and_save(input_file: str, output_file: str, n: int):
    try:
        # Carica i dati
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Applica la selezione
        result = select_round_robin_from_clusters(data, n)

        # Salva il risultato
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Risultato salvato in: {output_file}")

    except Exception as e:
        print(f"Errore: {e}")

