import json
import os
import sys
import shutil

from codebert_analyzer import CodeBERTAnalyzer
from extraction_classes.base_classes import Language
from extraction_classes.method_extractor import MultiLanguageMethodExtractor
from save_results import SaveResults
from method_selector import process_and_save

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

    # Prima di continuare elimino cartella analisi se esiste
    folder_path = 'analysis_output'
    file_path = 'selected_methods.json'
    print(f"Pulizia dati di precedenti computazioni.")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Cartella '{folder_path}' eliminata.")
    else:
        print(f"La cartella '{folder_path}' non esiste.")

    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' eliminato.")
    else:
        print(f"Il file '{file_path}' non esiste.")

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
    print("Inizio selezione metodi ...")
    process_and_save("analysis_output/complete_analysis.json", "selected_methods.json", num_elements)
    print("Selezione metodi completata")
    print("=" * 50)


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
