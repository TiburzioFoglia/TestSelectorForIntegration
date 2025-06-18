import json
from typing import Dict, Any


def select_round_robin_from_clusters(input_data: Dict[str, Any], n: int) -> Dict[str, Any]:
    # Estrai la sezione cluster_analysis
    cluster_analysis = input_data.get("detailed_results", {}).get("cluster_analysis", {})

    if not cluster_analysis:
        raise ValueError("Nessun cluster_analysis trovato nei dati")

    # Crea liste di metodi per ogni categoria (cluster + outliers)
    all_categories = {}

    for key, value in cluster_analysis.items():
        if "methods" in value:
            all_categories[key] = list(value["methods"])

    if not all_categories:
        raise ValueError("Nessun metodo trovato nei cluster o outliers")

    # Esegui selezione round-robin
    selected_methods = {}
    category_names = list(all_categories.keys())
    category_indices = {name: 0 for name in category_names}

    # Inizializza le liste per ogni categoria
    for category in category_names:
        selected_methods[category] = []

    selected_count = 0
    round_index = 0

    # Round-robin selection
    while selected_count < n:
        # Determina la categoria corrente
        current_category = category_names[round_index % len(category_names)]

        # Verifica se la categoria ha ancora elementi
        if category_indices[current_category] < len(all_categories[current_category]):
            method = all_categories[current_category][category_indices[current_category]]
            selected_methods[current_category].append(method)
            category_indices[current_category] += 1
            selected_count += 1

        round_index += 1

        # Se tutte le categorie sono esaurite, esci
        if all(category_indices[cat] >= len(all_categories[cat]) for cat in category_names):
            break

    # Costruisci il risultato finale - solo cluster con metodi selezionati
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

