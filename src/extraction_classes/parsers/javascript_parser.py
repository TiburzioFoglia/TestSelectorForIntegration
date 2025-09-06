import re
from typing import List, Optional, Dict, Any
from src.extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class JavaScriptParser(LanguageParser):
    """Parser specifico per il linguaggio JavaScript"""

    def get_method_pattern(self) -> str:
        # Pattern complesso per gestire vari tipi di funzioni JS
        return r'''
            (?P<modifiers>(?:async\s+|static\s+)*)\s*        # Modificatori opzionali
            (?:
                function\s+(?P<func_name>\w+)|               # Function declaration
                (?P<method_name>\w+)\s*(?::\s*function)?\s*| # Method/property
                (?P<arrow_name>\w+)\s*=\s*(?:async\s+)?      # Arrow function
            )
            \((?P<parameters>.*?)\)\s*                       # Parametri
            (?:=>\s*)?                                       # Arrow function =>
            (?P<body>\{(?:[^{}]|(?:\{(?:[^{}]|\{[^}]*\})*\}))*\})  # Corpo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'describe\(',
            r'it\(',
            r'test\(',
            r'@Test',
            r'jest\.',
            r'mocha\.'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            modifiers = match.group('modifiers') or ''
            method_name = (match.group('func_name') or
                           match.group('method_name') or
                           match.group('arrow_name'))

            if not method_name:
                return None

            parameters = match.group('parameters')
            method_body = match.group(0)

            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # JavaScript non ha attributi formali, ma possiamo cercare JSDoc
            attributes = []
            jsdoc_match = re.search(r'/\*\*(.*?)\*/', content[:start_pos][::-1][:200], re.DOTALL)
            if jsdoc_match:
                attributes.append('/**...*/')

            # Aggiungi modificatori come attributi
            if 'async' in modifiers:
                attributes.append('async')
            if 'static' in modifiers:
                attributes.append('static')

            # JavaScript è per default pubblico
            access_modifier = 'public'

            return MethodInfo(
                name=method_name,
                body=method_body.strip(),
                attributes=attributes,
                access_modifier=access_modifier,
                line_start=line_start,
                line_end=line_end,
                language=Language.JAVASCRIPT,
                parameters=parameters,
                return_type=""
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo JavaScript: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            'jest.mock(',
            'jest.fn(',
            '.mockReturnValue(',
            '.mockResolvedValue(',
            '.mockImplementation(',
            'sinon.stub(',
            'sinon.mock(',
            '.calledWith(',
            '.verify(',
        ]

    def get_comments_config(self) -> Dict[str, Any]:
        return {
            'comment_symbol': '//',
            'method_regex': re.compile(r'^\s*(async\s+)?(function\s+)?(?P<method_name>[a-zA-Z_0-9]+)\s*\(')
        }