import re
from typing import List, Optional
from extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class PythonParser(LanguageParser):
    """Parser specifico per il linguaggio Python"""

    def get_method_pattern(self) -> str:
        return r'''
            (?P<decorators>(?:\s*@\w+(?:\(.*?\))?\s*)*)\s*  # Decorators opzionali
            (?P<modifiers>(?:async\s+)?)\s*                  # async opzionale
            def\s+(?P<method_name>\w+)\s*                    # Nome metodo
            \((?P<parameters>.*?)\)\s*                       # Parametri
            (?::\s*(?P<return_type>[^:]+?))?\s*              # Type hint opzionale
            (?P<body>:(?:\n(?:[ \t]+.*(?:\n|$))+|\s*\n))     # Corpo del metodo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'@pytest\.mark\.',
            r'@unittest\.',
            r'def test_'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            decorators_str = match.group('decorators').strip()
            modifiers = match.group('modifiers') or ''
            method_name = match.group('method_name')
            return_type = match.group('return_type') or ''
            parameters = match.group('parameters')
            method_body = match.group(0)

            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Estrai decoratori
            attributes = []
            if decorators_str:
                decorators = re.findall(r'@(\w+(?:\([^)]*\))?)', decorators_str)
                attributes = [f'@{dec}' for dec in decorators]

            # Determina access modifier basato su convenzioni Python
            access_modifier = 'public'
            if method_name.startswith('__'):
                access_modifier = 'private'
            elif method_name.startswith('_'):
                access_modifier = 'protected'

            return MethodInfo(
                name=method_name,
                body=method_body.strip(),
                attributes=attributes,
                access_modifier=access_modifier,
                line_start=line_start,
                line_end=line_end,
                language=Language.PYTHON,
                parameters=parameters,
                return_type=return_type.strip() if return_type else ""
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo Python: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            'unittest.mock.Mock(',
            'unittest.mock.patch(',
            'mock.patch(',
            '@patch(',
            'mock.Mock(',
            '.assert_called_with(',
            '.return_value',
            '.side_effect',
            '.assert_called_once_with(',
        ]
