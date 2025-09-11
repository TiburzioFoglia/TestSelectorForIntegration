import re
from typing import List, Optional, Dict, Any
from src.extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class RubyParser(LanguageParser):
    """Parser specifico per il linguaggio Ruby"""

    def get_method_pattern(self) -> str:
        return r'''
            (?P<modifiers>(?:private|protected|public)\s+)?\s*  # Modificatori opzionali
            def\s+(?P<method_name>\w+[?!]?)\s*               # Nome metodo (con ? o !)
            (?:\((?P<parameters>.*?)\))?\s*                  # Parametri opzionali
            (?P<body>.*?)                                    # Corpo
            \s*end\b                                         # Fine metodo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'def test_',
            r'it\s+["\']',
            r'describe\s+["\']',
            r'context\s+["\']',
            r'should\s+["\']'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            modifiers = match.group('modifiers') or ''
            method_name = match.group('method_name')
            parameters = match.group('parameters') or ''
            method_body = match.group(0)

            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Ruby usa commenti per documentazione
            attributes = []

            # Cerca commenti di documentazione prima del metodo
            method_start_line = line_start - 1
            comment_lines = []

            # Guarda le righe precedenti per commenti
            for i in range(max(0, method_start_line - 10), method_start_line):
                if i < len(lines) and lines[i].strip().startswith('#'):
                    comment_lines.append(lines[i].strip())
                elif lines[i].strip() == '':
                    continue  # Ignora righe vuote
                else:
                    break  # Se trova una riga non vuota e non commento, ferma la ricerca

            if comment_lines:
                attributes.append('# documented')

            # Rileva tipo di metodo da suffissi Ruby
            if method_name.endswith('?'):
                attributes.append('predicate')
            elif method_name.endswith('!'):
                attributes.append('mutating')

            # Determina il modificatore di accesso
            access_modifier = 'public'  # default Ruby
            if 'private' in modifiers:
                access_modifier = 'private'
            elif 'protected' in modifiers:
                access_modifier = 'protected'

            return MethodInfo(
                name=method_name,
                body=method_body.strip(),
                attributes=attributes,
                access_modifier=access_modifier,
                line_start=line_start,
                line_end=line_end,
                language=Language.RUBY,
                parameters=parameters,
                return_type=""
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo Ruby: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            'allow(',
            'expect(',
            '.to receive(',
            '.and_return(',
            'mock(',
            'stub(',
            'stub_request(',
        ]

    def get_comments_config(self) -> Dict[str, Any]:
        return {
        'comment_symbol': '#',
        'method_regex': re.compile(r'^\s*def\s+(?P<method_name>[a-zA-Z_0-9_!?=]+)')
        }

    def line_contains_mock(self, line: str) -> bool:
        """Controlla se la riga matcha uno qualsiasi dei filtri di mocking."""
        return None

    def extract_mock_subject(self, line: str) -> Optional[str]:
        """Estrae il soggetto del mock usando il pattern specifico per Python."""
        return None
