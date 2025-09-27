import argparse
import json
import os
import sys
import shutil

from src.analysis_classes.code_analyzer import ComprehensiveCodeAnalyzer
from src.extraction_classes.base_classes import Language
from src.extraction_classes.method_extractor import MultiLanguageMethodExtractor
from src.method_selector import process_and_save

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

    ANALYZERS = ["codeBert", "graphCodeBert", "codeT5", "polyCoder", "sentenceTransformer", "unixCoder", "starCoder2"]

    parser = argparse.ArgumentParser(
        description="Run automated tests selection on a tests file."
    )
    parser.add_argument("file_name", help="Target tests file")
    parser.add_argument(
        "n_test",
        type=int,
        nargs="?",
        default=10,
        help="Number of tests to select (default: 10)"
    )
    parser.add_argument(
        "--code-analyzer",
        choices=ANALYZERS,
        default="codeBert",
        help=f"Choose a code analyzer (default: codeBert, options: {', '.join(ANALYZERS)})"
    )
    parser.add_argument(
        "--full-tests",
        action="store_true",
        help="Run tests selection taking into consideration the full tests"
    )

    args = parser.parse_args()

    file_name = args.file_name
    num_elements = args.n_test
    analyzer = args.code_analyzer
    full_tests = args.full_tests

    print("=" * 50)
    print("File:", file_name)
    print("N Tests:", num_elements)
    print(f"Analyzer: {analyzer}")
    print("Mocks only:", not full_tests)

    language = get_language(args.file_name)

    print(f"Linguaggio selezionato: {language.value}")
    print("=" * 50)

    # Prima di continuare elimino cartella analisi se esiste
    folder_path = 'analysis_output'
    print(f"Pulizia dati di precedenti computazioni.")

    comprehensive_analysis_path = "analysis_output/comprehensive_analysis_results.json"
    selected_methods_path = "analysis_output/selected_methods.json"

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Cartella '{folder_path}' eliminata.")
    else:
        print(f"La cartella '{folder_path}' non esiste.")

    print("=" * 50)

    methods_info = process_file(file_name, language, folder_path, full_tests)
    if methods_info is None:
        print("Something went wrong")
        sys.exit(1)
    print("=" * 50)


    print("Inizio analisi metodi ...")
    if not full_tests:
        cluster_analysis(os.path.join(folder_path, 'extracted_methods.json'), folder_path, analyzer,
                         os.path.join(folder_path, 'complete_extracted_methods.json'))
    else:
        cluster_analysis(os.path.join(folder_path, 'extracted_methods.json'), folder_path, analyzer)

    print("=" * 50)
    print("Inizio selezione metodi ...")
    process_and_save(comprehensive_analysis_path, selected_methods_path, num_elements)
    print("Selezione metodi completata")
    print("=" * 50)


    print("Inizio creazione file con metodi commentati ...")
    create_commented_copy(file_name, comprehensive_analysis_path, selected_methods_path, language)
    print("=" * 50)

    if not full_tests:
        selected_methods_file = os.path.join(folder_path, 'selected_methods.json')
        extracted_methods_file = os.path.join(folder_path, 'extracted_methods.json')

        cip_result = calculate_cip(selected_methods_file, extracted_methods_file)

        print(f"Risultato della Coverage of Integration Points (CIP): {cip_result:.2f}%")
        print("=" * 50)


def get_language(file_name):

    ext = os.path.splitext(file_name)[1].lower()
    lang = LANGUAGES.get(ext)
    if lang is None:
        print(f"[ERRORE] Estensione file '{ext}' non riconosciuta.")
        sys.exit(1)
    return lang


def process_file(file_path: str,language: Language, output_dir: str = "analysis_output", full_tests: bool = False):
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

    print("Eliminando dai metodi elementi non mock...")

    if not full_tests:
        filtered_test_methods = extractor_instance.strip_methods_to_mock_lines(test_methods_with_mocks)
    else:
        filtered_test_methods = test_methods_with_mocks

    if not filtered_test_methods:
        print("Qualcosa è andato storto nella creazione di metodi con solo mock")
        return None, None
    else:
        print(f"Ottenuti {len(filtered_test_methods)} metodi di test con solo mock")


    print("Preparazione per l'embedder...")
    clean_methods = extractor_instance.methods_to_codebert_format(filtered_test_methods)
    methods_info = create_extracted_methods_file(filtered_test_methods, clean_methods,
                                                 output_dir, "extracted_methods.json")

    if not full_tests:
        complete_clean_methods = extractor_instance.methods_to_codebert_format(test_methods_with_mocks)
        create_extracted_methods_file(filtered_test_methods, complete_clean_methods,
                                      output_dir, "complete_extracted_methods.json")

    return methods_info

def create_extracted_methods_file(filtered_test_methods: list, clean_methods: list, output_dir: str, file_name: str):
    if len(filtered_test_methods) != len(clean_methods):
        raise ValueError("Le liste di informazioni sui metodi e dei corpi non hanno la stessa lunghezza!")

    methods_info = [
        {"name": method_info.name, "code": body}
        for method_info, body in zip(filtered_test_methods, clean_methods)
    ]

    methods_file = os.path.join(output_dir, file_name)
    with open(methods_file, 'w', encoding='utf-8') as f:
        json.dump(methods_info, f, indent=2, ensure_ascii=False)

    print(f"Salvati metodi per l'embedder come {file_name}")
    return methods_info

def cluster_analysis(input_file: str, output_dir: str, analyzer: str, complete_methods_file: str = None):
    # Inizializza l'analyzer
    analyzer = ComprehensiveCodeAnalyzer(analyzer)

    if complete_methods_file is not None:
        results = analyzer.analyze_extracted_methods(input_file,output_dir, complete_methods_file)
    else:
        results = analyzer.analyze_extracted_methods(input_file,output_dir)

    print("\nAnalisi completata con successo!")
    return results

def create_commented_copy(source_file_path: str, analysis_results_path: str, selected_methods_path: str,
        language: Language, suffix: str = '_commented'):

    try:
        extractor_instance = MultiLanguageMethodExtractor()
        extractor_instance.set_language(language)
        config = extractor_instance.current_parser.get_comments_config()
        comment_symbol = config['comment_symbol']
        method_regex = config['method_regex']

        with open(analysis_results_path, 'r') as f:
            analysis_data = json.load(f)
        with open(selected_methods_path, 'r') as f:
            selected_data = json.load(f)
        with open(source_file_path, 'r') as f:
            source_lines = f.readlines()

        method_info = {m['method_name']: m for m in analysis_data.get('method_analysis', [])}
        selected_methods = {m for d in selected_data.values() for m in d.get('methods', [])}
        methods_to_comment = set(method_info.keys()) - selected_methods

        if not methods_to_comment:
            print(f"Info: Nessun metodo da commentare. Nessuna copia creata per '{source_file_path}'.")
            return None

        modified_lines = []
        i = 0
        while i < len(source_lines):
            line = source_lines[i]
            match = method_regex.search(line)

            if match and (method_name := match.group('method_name')) in methods_to_comment:
                lines_to_find = method_info[method_name].get('lines_of_code', 1)
                code_lines_found = 0

                current_pos = i
                while code_lines_found < lines_to_find and current_pos < len(source_lines):
                    current_line = source_lines[current_pos]
                    stripped_line = current_line.strip()

                    if not stripped_line:
                        modified_lines.append(current_line)  # Preserva le righe vuote per leggibilità
                    else:
                        modified_lines.append(f"{comment_symbol} {current_line.rstrip()}\n")

                    # Ma incrementiamo il contatore solo se è codice
                    is_comment = stripped_line.startswith(comment_symbol)
                    is_code = not is_comment and stripped_line

                    if is_code:
                        code_lines_found += 1

                    current_pos += 1

                i = current_pos
                continue  # Passa alla prossima iterazione del ciclo while

            # Se la riga non è l'inizio di un metodo da commentare, aggiungila normalmente
            modified_lines.append(line)
            i += 1

        modified_code = "".join(modified_lines)

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Errore durante l'elaborazione dei file: {e}")
        return None

    try:
        base_name, extension = os.path.splitext(source_file_path)

        output_file_path = f"{base_name}{suffix}{extension}"

        with open(output_file_path, 'w') as f:
            f.write(modified_code)

        print(f"Successo! File {output_file_path} creato.")

    except IOError as e:
        print(f"Errore durante il salvataggio del file: {e}")
        return None


def calculate_cip(selected_methods_path: str, extracted_methods_path: str) -> float:

    with open(selected_methods_path, 'r') as f:
        selected_methods_data = json.load(f)

    with open(extracted_methods_path, 'r') as f:
        extracted_methods_data = json.load(f)

    # Trova tutti i punti di integrazione unici (Total_Interactions)
    total_interactions = set()
    extracted_methods_map = {method['name']: method['code'] for method in extracted_methods_data}

    for method_code in extracted_methods_map.values():
        interactions = method_code.strip().split('\n')
        total_interactions.update(interaction for interaction in interactions if interaction)

    # Ottieni la lista di tutti i metodi selezionati
    selected_method_names = set()
    for cluster_info in selected_methods_data.values():
        selected_method_names.update(cluster_info['methods'])

    # Trova i punti di integrazione unici coperti dai metodi selezionati (Selected_Interactions)
    selected_interactions = set()
    for method_name in selected_method_names:
        if method_name in extracted_methods_map:
            method_code = extracted_methods_map[method_name]
            interactions = method_code.strip().split('\n')
            selected_interactions.update(interaction for interaction in interactions if interaction)

    # Calcola la CIP usando la formula
    num_total_interactions = len(total_interactions)
    num_selected_interactions = len(selected_interactions)

    print(f"Numero di punti di integrazione unici totali: {num_total_interactions}")
    # print(f"Punti di integrazione totali: {sorted(list(total_interactions))}\n")
    print(f"Numero di punti di integrazione unici selezionati: {num_selected_interactions}")
    # print(f"Punti di integrazione selezionati: {sorted(list(selected_interactions))}\n")

    if num_total_interactions == 0:
        return 0.0

    cip = (num_selected_interactions / num_total_interactions) * 100

    return cip


if __name__ == "__main__":
    main()
    # folder_path = 'analysis_output'
    # selected_methods_file = os.path.join(folder_path, 'selected_methods.json')
    # extracted_methods_file = 'extracted_methods_owner.json'
    #
    # cpi_result = calculate_cpi(selected_methods_file, extracted_methods_file)
    #
    # print(f"Risultato della Coverage of Integration Points (CPI): {cpi_result:.2f}%")