import json
import os
import sys
from itertools import cycle

from codebert_analyzer import CodeBERTAnalyzer
from extraction_classes.base_classes import Language
from extraction_classes.method_extractor import MultiLanguageMethodExtractor
from save_results import SaveResults

LANGUAGES = {
    '.cs': Language.CSHARP,
    '.py': Language.PYTHON,
    '.java': Language.JAVA,
    '.js': Language.JAVASCRIPT,
    '.php': Language.PHP,
    '.rb': Language.RUBY,
    '.go': Language.GO
}

def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py file_name numero_test [linguaggio_opzionale]")
        sys.exit(1)

    file_name = sys.argv[1]
    num_elements = int(sys.argv[2])
    lang_arg = sys.argv[3] if len(sys.argv) > 3 else None

    language = get_language(file_name, lang_arg)

    print("=" * 50)
    print(f"File: {file_name}")
    print(f"Numero elementi: {num_elements}")
    print(f"Linguaggio selezionato: {language.value}")
    print("=" * 50)

    methods, clean_methods = process_file(file_name, language)
    if (methods is None) or (clean_methods is None):
        print("Something went wrong")
        sys.exit(1)
    print("=" * 50)

    results = codebert_analysis(methods,clean_methods)
    if results:
        print("\nAnalisi completata con successo!")
    else:
        print("Analisi fallita")

    print("=" * 50)

    round_robin_selection(num_elements)
    print("")
    print("=" * 50)
    #aggiungere metodo per scelta automatica

# TODO fix
def round_robin_selection(n):

    with open("analysis_output/complete_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cluster_data = data.get("detailed_results", {}).get("cluster_analysis", {})

    selected = {k: {"method_count": 0, "avg_complexity": 0.0, "avg_lines": 0.0,
                    "methods": [], "characteristics": v["characteristics"]}
                for k, v in cluster_data.items()}

    clusters = [k for k in cluster_data if cluster_data[k]["methods"]]
    cluster_cycle = cycle(clusters)
    method_count = 0

    while method_count < n and clusters:
        current = next(cluster_cycle)
        if cluster_data[current]["methods"]:
            method = cluster_data[current]["methods"].pop(0)
            selected[current]["methods"].append(method)
            method_count += 1
        # Rimuovi cluster vuoti dal ciclo
        clusters = [c for c in clusters if cluster_data[c]["methods"]]
        cluster_cycle = cycle(clusters)

    clusters_to_delete = []
    for k in selected:
        methods = selected[k]["methods"]
        count = len(methods)
        if count > 0:
            orig = cluster_data[k]
            selected[k]["method_count"] = count
            # selected[k]["avg_complexity"] = orig["avg_complexity"]
            # selected[k]["avg_lines"] = orig["avg_lines"]
        else:
            clusters_to_delete.append(k)

    # Rimuove i cluster vuoti dopo l’iterazione
    for k in clusters_to_delete:
        del selected[k]

    output_data = {
        "selected_methods": {
            "cluster_analysis": selected
        }
    }

    with open("selected_methods.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def get_language(file_name, lang_arg=None):
    if lang_arg:
        lang_str = lang_arg.lower()
        for lang in Language:
            if lang.value == lang_str:
                return lang
        print(f"[ERRORE] Linguaggio non supportato: '{lang_arg}'")
        sys.exit(1)
    else:
        ext = os.path.splitext(file_name)[1].lower()
        lang = LANGUAGES.get(ext)
        if lang is None:
            print(f"[ERRORE] Estensione file '{ext}' non riconosciuta.")
            sys.exit(1)
        return lang


def process_file(file_path: str,language: Language, output_dir: str = "analysis_output"):
    print(f"Processando file: {file_path}")

    if not os.path.exists(file_path):
        print(f"File non trovato: {file_path}")
        return None, None

    os.makedirs(output_dir, exist_ok=True)

    print("Estrazione metodi...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    extractor_instance = MultiLanguageMethodExtractor()
    extractor_instance.set_language(language)

    test_methods = extractor_instance.extract_test_methods_only(content)

    if not test_methods:
        print("Nessun metodo di test trovato nel file")
        return None, None
    else:
        print(f"Identificati {len(test_methods)} metodi di test")

    print("Filtrando metodi con mock...")

    test_methods_with_mocks = extractor_instance.filter_for_mocks(test_methods)

    if not test_methods_with_mocks:
        print("Nessun metodo di test trovato nel file")
        return None, None
    else:
        print(f"Identificati {len(test_methods_with_mocks)} metodi di test con mock")


    print("Preparazione per CodeBERT...")
    clean_methods = extractor_instance.methods_to_codebert_format(test_methods_with_mocks)

    methods_file = os.path.join(output_dir, "extracted_methods.json")
    with open(methods_file, 'w', encoding='utf-8') as f:
        json.dump(clean_methods, f, indent=2, ensure_ascii=False)
    print("Salvati metodi per CodeBERT come extracted_methods.json")
    return test_methods_with_mocks, clean_methods


def codebert_analysis(methods: list,clean_methods: list, output_dir: str = "analysis_output"):
    print("Inizializzazione CodeBERT...")
    codebert_analyzer = CodeBERTAnalyzer()

    print("Calcolo embeddings...")
    embeddings = codebert_analyzer.get_code_embeddings(clean_methods)

    methods_info = []
    for i, method in enumerate(methods):
        methods_info.append({
            'name': method.name,
            'body': method.body,
            'attributes': method.attributes,
            'access_modifier': method.access_modifier,
            'line_start': method.line_start,
            'line_end': method.line_end,
            'clean_body': clean_methods[i]
        })

    # Genera analisi completa
    print("Generazione report di analisi...")
    results = codebert_analyzer.generate_analysis_report(methods_info, embeddings)

    # Salva tutto nella directory di output
    results_saver = SaveResults()
    results_saver.save_complete_results(results, methods_info, output_dir)

    return results




if __name__ == "__main__":
    main()
