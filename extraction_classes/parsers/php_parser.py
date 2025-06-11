import re
from typing import List, Optional
from extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class PHPParser(LanguageParser):
    """Parser specifico per il linguaggio PHP"""

    def get_method_pattern(self) -> str:
        return r'''
            (?P<modifiers>(?:public|private|protected|static|final|abstract)\s+)*  # Modificatori
            function\s+(?P<method_name>\w+)\s*               # Nome metodo
            \((?P<parameters>.*?)\)\s*                       # Parametri
            (?::\s*(?P<return_type>\??[\w\\]+))?\s*          # Type hint opzionale
            (?P<body>\{(?:[^{}]|(?:\{(?:[^{}]|\{[^}]*\})*\}))*\})  # Corpo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'@test',
            r'@Test',
            r'function test',
            r'@dataProvider',
            r'@depends'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            modifiers = match.group('modifiers') or ''
            method_name = match.group('method_name')
            return_type = match.group('return_type') or ''
            parameters = match.group('parameters')
            method_body = match.group(0)

            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Cerca PHPDoc prima del metodo
            attributes = []
            phpdoc_match = re.search(r'/\*\*(.*?)\*/', content[:start_pos][::-1][:500], re.DOTALL)
            if phpdoc_match:
                # Estrai annotazioni PHPDoc
                phpdoc_content = phpdoc_match.group(1)[::-1]  # Inverte di nuovo
                annotations = re.findall(r'@(\w+)', phpdoc_content)
                attributes.extend([f'@{ann}' for ann in annotations])

            # Aggiungi modificatori come attributi se presenti
            if 'static' in modifiers:
                attributes.append('static')
            if 'final' in modifiers:
                attributes.append('final')
            if 'abstract' in modifiers:
                attributes.append('abstract')

            # Determina il modificatore di accesso
            access_modifier = 'public'  # default PHP
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
                language=Language.PHP,
                parameters=parameters,
                return_type=return_type
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo PHP: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            '$this->createMock(',
            '$this->getMockBuilder(',
            '->expects(',
            '->method(',
            'Mockery::mock(',
            '->shouldReceive(',
            '->andReturn(',
            '->once()',
            '->times(',
        ]
