import re
from typing import List, Dict
from copy import deepcopy

from src.extraction_classes.base_classes import Language, MethodInfo

# Import dei parser specifici
from .parsers.csharp_parser import CSharpParser
from .parsers.python_parser import PythonParser
from .parsers.java_parser import JavaParser
from .parsers.javascript_parser import JavaScriptParser
from .parsers.php_parser import PHPParser
from .parsers.ruby_parser import RubyParser
from .parsers.go_parser import GoParser


class MultiLanguageMethodExtractor:
    """
    Estrattore di metodi multi-linguaggio che supporta:
    C#, Python, Java, JavaScript, PHP, Ruby, Go
    """

    def __init__(self, language: Language = Language.CSHARP):
        self.parsers = {
            Language.CSHARP: CSharpParser(),
            Language.PYTHON: PythonParser(),
            Language.JAVA: JavaParser(),
            Language.JAVASCRIPT: JavaScriptParser(),
            Language.PHP: PHPParser(),
            Language.RUBY: RubyParser(),
            Language.GO: GoParser()
        }
        self.current_language = language
        self.current_parser = self.parsers[language]

    def set_language(self, language: Language) -> None:
        if language not in self.parsers:
            raise ValueError(f"Linguaggio {language} non supportato")

        self.current_language = language
        self.current_parser = self.parsers[language]

    def extract_methods_from_content(self, content: str) -> List[MethodInfo]:
        if not content.strip():
            return []

        # Python usa AST, gli altri usano regex.
        if self.current_language == Language.PYTHON:
            try:
                # Usa il metodo speciale basato su AST per Python
                return self.current_parser.extract_methods_with_ast(content)
            except Exception as e:
                print(f"Errore durante l'estrazione AST per {self.current_language}: {e}")
                return []

        methods = []
        lines = content.split('\n')

        try:
            pattern = re.compile(
                self.current_parser.get_method_pattern(),
                re.VERBOSE | re.DOTALL | re.MULTILINE
            )
            matches = pattern.finditer(content)

            for match in matches:
                method_info = self.current_parser.parse_method_match(match, content, lines)
                if method_info:
                    methods.append(method_info)

        except Exception as e:
            print(f"Errore durante l'estrazione per {self.current_language}: {e}")

        return methods

    def extract_test_methods_only(self, content: str) -> List[MethodInfo]:
        """Estrae solo i metodi di test nel linguaggio corrente"""
        all_methods = self.extract_methods_from_content(content)
        test_methods = []

        for method in all_methods:
            if self._is_test_method(method):
                test_methods.append(method)

        return test_methods

    def _is_test_method(self, method: MethodInfo) -> bool:
        """Verifica se un metodo è un test basandosi sui suoi attributi e nome"""
        parser = self.parsers[method.language]
        test_patterns = parser.get_test_attributes()

        # Controlla attributi/decoratori
        for attr in method.attributes:
            for test_pattern in test_patterns:
                if re.search(test_pattern, attr, re.IGNORECASE):
                    return True

        # Per python, vogliamo anche controllare i nomi che iniziano con 'test_'
        if method.language == Language.PYTHON and method.name.startswith("test_"):
            return True

        # Controlla nome del metodo
        for test_pattern in test_patterns:
            if re.search(test_pattern, method.name, re.IGNORECASE):
                return True

        return False

    def methods_to_codebert_format(self, methods: List[MethodInfo]) -> List[str]:
        """Converte i metodi in formato adatto per CodeBERT"""
        codebert_ready = []

        for method in methods:
            clean_body = self._clean_code_for_bert(method.body)
            codebert_ready.append(clean_body)

        return codebert_ready

    def _clean_code_for_bert(self, code: str) -> str:
        """Pulisce il codice per l'analisi con CodeBERT"""
        # Rimuovi commenti (per tutti i linguaggi)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # C-style
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python/Ruby
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line

        # Normalizza whitespace
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'^\s+', '', code, flags=re.MULTILINE)

        return code.strip()

    def get_supported_languages(self) -> List[Language]:
        """Restituisce la lista dei linguaggi supportati"""
        return list(self.parsers.keys())

    def get_language_stats(self, methods: List[MethodInfo]) -> Dict[Language, int]:
        """Restituisce statistiche sui metodi per linguaggio"""
        stats = {}
        for method in methods:
            if method.language in stats:
                stats[method.language] += 1
            else:
                stats[method.language] = 1
        return stats

    def filter_for_mocks(self, methods: List[MethodInfo]) -> List[MethodInfo]:
        filters = self.current_parser.get_mock_filters()
        filtered_methods = []

        if self.current_language == Language.PYTHON:
            # Compiliamo le regex una sola volta per efficienza
            compiled_filters = [re.compile(f) for f in filters]

            for method in methods:
                # Usiamo re.search per trovare corrispondenze nel corpo del metodo
                for pattern in compiled_filters:
                    if pattern.search(method.body):
                        filtered_methods.append(method)
                        break  # Trovato un match, passiamo al metodo successivo

            return filtered_methods

        for method in methods:
            if any(f in method.body for f in filters):
                filtered_methods.append(method)

        return filtered_methods

    def strip_methods_to_mock_lines(self, methods: List[MethodInfo]) -> List[MethodInfo]:
        stripped_method_info_list = []
        parser = self.current_parser

        for method in methods:
            lines = method.body.split('\n')
            extracted_mocks = []

            for line in lines:
                # Chiedi al parser se la riga è rilevante
                if parser.line_contains_mock(line):
                    # Se sì, estrae il soggetto
                    mock_subject = parser.extract_mock_subject(line)
                    if mock_subject:
                        extracted_mocks.append(mock_subject)

            # Crea una copia profonda e aggiorna il corpo del metodo
            new_method_info = deepcopy(method)
            new_method_info.body = "\n".join(extracted_mocks)
            stripped_method_info_list.append(new_method_info)

        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(stripped_method_info_list)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        return stripped_method_info_list

    # def strip_methods_to_mock_lines(self, methods: List[MethodInfo]) -> List[MethodInfo]:
    #     stripped_method_info_list = []
    #     parser = self.current_parser
    #
    #     for method in methods:
    #
    #         # Otteniamo i filtri (stringhe o regex) direttamente dal parser.
    #         mock_filters = parser.get_mock_filters()
    #
    #         mock_lines = []
    #         if mock_filters:
    #             lines = method.body.split('\n')
    #
    #             if method.language == Language.PYTHON:
    #                 compiled_filters = [re.compile(f) for f in mock_filters]
    #                 for line in lines:
    #                     if any(p.search(line) for p in compiled_filters):
    #                         mock_lines.append(line)
    #             else:
    #                 for line in lines:
    #                     if any(f in line for f in mock_filters):
    #                         mock_lines.append(line)
    #
    #         # Crea una copia profonda per non modificare l'oggetto originale
    #         new_method_info = deepcopy(method)
    #         # Sostituisci il body con le sole righe di mocking
    #         new_method_info.body = "\n".join(mock_lines).strip()
    #
    #         stripped_method_info_list.append(new_method_info)
    #
    #     return stripped_method_info_list
