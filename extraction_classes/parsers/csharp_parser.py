import re
from typing import List, Optional
from extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class CSharpParser(LanguageParser):
    """Parser specifico per il linguaggio C#"""

    def get_method_pattern(self) -> str:
        return r'''
            (?P<attributes>(?:\s*\[.*?\]\s*)*)\s*         # Attributi opzionali
            (?P<modifiers>(?:public|private|protected|internal|static|virtual|override|async)\s+)*  # Modificatori
            (?P<return_type>\w+(?:<.*?>)?|\w+\[\])\s+     # Tipo di ritorno
            (?P<method_name>\w+)\s*                       # Nome metodo
            \((?P<parameters>.*?)\)\s*                    # Parametri
            (?P<body>\{(?:[^{}]|(?:\{(?:[^{}]|\{[^}]*\})*\}))*\})  # Corpo del metodo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'\[Test\]',
            r'\[TestMethod\]',
            r'\[Fact\]',
            r'\[Theory\]',
            r'\[TestCase.*?\]'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            attributes_str = match.group('attributes').strip()
            modifiers = match.group('modifiers') or ''
            method_name = match.group('method_name')
            return_type = match.group('return_type')
            parameters = match.group('parameters')
            method_body = match.group(0)

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
                line_end=line_end,
                language=Language.CSHARP,
                parameters=parameters,
                return_type=return_type
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo C#: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            'new Mock<',
            '.Setup(',
            '.Verify(',
            '.Returns(',
            '.Throws(',
            'It.IsAny<',
            '.Object',
            '[Mock]',
            '.CallBase',
        ]