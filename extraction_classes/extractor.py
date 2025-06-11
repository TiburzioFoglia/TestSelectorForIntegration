import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class MethodInfo:
    name: str
    body: str
    attributes: List[str]
    access_modifier: str
    line_start: int
    line_end: int


class CSharpMethodExtractor:
    def __init__(self):
        # Pattern per identificare attributi di test comuni
        self.test_attributes = [
            r'\[Test\]',
            r'\[TestMethod\]',
            r'\[Fact\]',
            r'\[Theory\]',
            r'\[TestCase.*?\]'
        ]

        # Pattern per metodi
        self.method_pattern = r'''
            (?P<attributes>(?:\s*\[.*?\]\s*)*)\s*         # Attributi opzionali
            (?P<modifiers>(?:public|private|protected|internal|static|virtual|override|async)\s+)*  # Modificatori
            (?P<return_type>\w+(?:<.*?>)?|\w+\[\])\s+     # Tipo di ritorno
            (?P<method_name>\w+)\s*                       # Nome metodo
            \((?P<parameters>.*?)\)\s*                    # Parametri
            (?P<body>\{(?:[^{}]|(?:\{(?:[^{}]|\{[^}]*\})*\}))*\})  # Corpo del metodo
        '''

    # def extract_methods_from_file(self, file_path: str) -> List[MethodInfo]:
    #     """Estrae tutti i metodi da un file C#"""
    #     try:
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             content = file.read()
    #         return self.extract_methods_from_content(content)
    #     except Exception as e:
    #         print(f"Errore nella lettura del file {file_path}: {e}")
    #         return []

    def extract_methods_from_content(self, content: str) -> List[MethodInfo]:
        """Estrae tutti i metodi dal contenuto C#"""
        methods = []
        lines = content.split('\n')

        # Trova tutti i metodi usando regex
        pattern = re.compile(self.method_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.finditer(content)

        for match in matches:
            method_info = self._parse_method_match(match, content, lines)
            if method_info:
                methods.append(method_info)

        return methods

    def extract_test_methods_only(self, content: str) -> List[MethodInfo]:
        """Estrae solo i metodi di test (con attributi di test)"""
        all_methods = self.extract_methods_from_content(content)
        test_methods = []

        for method in all_methods:
            if self._is_test_method(method.attributes):
                test_methods.append(method)

        return test_methods

    def _parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        """Analizza una corrispondenza regex per creare MethodInfo"""
        try:
            attributes_str = match.group('attributes').strip()
            modifiers = match.group('modifiers') or ''
            method_name = match.group('method_name')
            method_body = match.group(0)  # Tutto il match

            # Trova le linee di inizio e fine
            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Estrai attributi
            attributes = []
            if attributes_str:
                attr_matches = re.findall(r'\[([^\]]+)\]', attributes_str)
                attributes = [f'[{attr}]' for attr in attr_matches]

            # Determina il modificatore di accesso
            access_modifier = 'public'  # default
            if 'private' in modifiers:
                access_modifier = 'private'
            elif 'protected' in modifiers:
                access_modifier = 'protected'
            elif 'internal' in modifiers:
                access_modifier = 'internal'

            return MethodInfo(
                name=method_name,
                body=method_body.strip(),
                attributes=attributes,
                access_modifier=access_modifier,
                line_start=line_start,
                line_end=line_end
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo: {e}")
            return None

    def _is_test_method(self, attributes: List[str]) -> bool:
        """Verifica se un metodo è un test basandosi sui suoi attributi"""
        for attr in attributes:
            for test_attr_pattern in self.test_attributes:
                if re.search(test_attr_pattern, attr, re.IGNORECASE):
                    return True
        return False

    # def save_methods_to_files(self, methods: List[MethodInfo], output_dir: str):
    #     """Salva ogni metodo in un file separato"""
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     for i, method in enumerate(methods):
    #         filename = f"{method.name}_{i}.cs"
    #         filepath = os.path.join(output_dir, filename)
    #
    #         with open(filepath, 'w', encoding='utf-8') as f:
    #             f.write(method.body)
    #
    #         print(f"Salvato: {filepath}")

    def methods_to_codebert_format(self, methods: List[MethodInfo]) -> List[str]:
        """Converte i metodi in formato adatto per CodeBERT"""
        codebert_ready = []

        for method in methods:
            # Rimuovi commenti e whitespace extra per CodeBERT
            clean_body = self._clean_code_for_bert(method.body)
            codebert_ready.append(clean_body)

        return codebert_ready

    def _clean_code_for_bert(self, code: str) -> str:
        """Pulisce il codice per l'analisi con CodeBERT"""
        # Rimuovi commenti su linea singola
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)

        # Rimuovi commenti multi-linea
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Normalizza whitespace
        code = re.sub(r'\n\s*\n', '\n', code)  # Rimuovi linee vuote multiple
        code = re.sub(r'^\s+', '', code, flags=re.MULTILINE)  # Rimuovi indentazione iniziale

        return code.strip()

