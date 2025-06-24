import re
from typing import List, Optional
from src.extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class JavaParser(LanguageParser):
    """Parser specifico per il linguaggio Java"""

    def get_method_pattern(self) -> str:
        return r'''
            (?P<annotations>(?:\s*@\w+(?:\(.*?\))?\s*)*)\s*  # Annotazioni opzionali
            (?P<modifiers>(?:public|private|protected|static|final|abstract|synchronized)\s+)*  # Modificatori
            (?P<return_type>(?:\w+(?:<.*?>)?|\w+\[\]|void))\s+  # Tipo di ritorno
            (?P<method_name>\w+)\s*                          # Nome metodo
            \((?P<parameters>.*?)\)\s*                       # Parametri
            (?:throws\s+[\w\s,]+)?\s*                        # Throws opzionale
            (?P<body>\{(?:[^{}]|(?:\{(?:[^{}]|\{[^}]*\})*\}))*\})  # Corpo del metodo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'@Test',
            r'@TestMethod',
            r'@ParameterizedTest',
            r'@RepeatedTest',
            r'@DisplayName'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            annotations_str = match.group('annotations').strip()
            modifiers = match.group('modifiers') or ''
            method_name = match.group('method_name')
            return_type = match.group('return_type')
            parameters = match.group('parameters')
            method_body = match.group(0)

            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Estrai annotazioni
            attributes = []
            if annotations_str:
                annotations = re.findall(r'@(\w+(?:\([^)]*\))?)', annotations_str)
                attributes = [f'@{ann}' for ann in annotations]

            # Determina il modificatore di accesso
            access_modifier = 'package'  # default Java
            if 'public' in modifiers:
                access_modifier = 'public'
            elif 'private' in modifiers:
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
                language=Language.JAVA,
                parameters=parameters,
                return_type=return_type
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo Java: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            '.mock(',
            '@Mock',
            '.when(',
            'verify(',
            'doReturn(',
            '.thenReturn(',
            'mock.',
            'Mock.',
            'Mock<'
        ]