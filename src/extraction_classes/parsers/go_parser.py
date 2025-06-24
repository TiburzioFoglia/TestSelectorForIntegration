from typing import List, Optional
from src.extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class GoParser(LanguageParser):
    """Parser specifico per il linguaggio Go"""

    def get_method_pattern(self) -> str:
        return r'''
            func\s+                                          # Keyword func
            (?:\((?P<receiver>.*?)\)\s+)?                    # Receiver opzionale
            (?P<method_name>\w+)\s*                          # Nome funzione
            \((?P<parameters>.*?)\)\s*                       # Parametri
            (?P<return_type>\(.*?\)|[\w\*\[\]]+)?\s*         # Tipo di ritorno
            (?P<body>\{(?:[^{}]|(?:\{(?:[^{}]|\{[^}]*\})*\}))*\})  # Corpo
        '''

    def get_test_attributes(self) -> List[str]:
        return [
            r'func Test',
            r'func Benchmark',
            r'func Example'
        ]

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        try:
            receiver = match.group('receiver') or ''
            method_name = match.group('method_name')
            return_type = match.group('return_type') or ''
            parameters = match.group('parameters')
            method_body = match.group(0)

            start_pos = match.start()
            end_pos = match.end()
            line_start = content[:start_pos].count('\n') + 1
            line_end = content[:end_pos].count('\n') + 1

            # Gestisci attributi specifici di Go
            attributes = []

            # Se ha un receiver, è un metodo
            if receiver:
                attributes.append(f'receiver: {receiver.strip()}')

            # Cerca commenti di documentazione Go (// prima della funzione)
            method_start_line = line_start - 1
            comment_lines = []

            for i in range(max(0, method_start_line - 5), method_start_line):
                if i < len(lines):
                    line = lines[i].strip()
                    if line.startswith('//'):
                        comment_lines.append(line)
                    elif line == '':
                        continue
                    else:
                        break

            if comment_lines:
                attributes.append('// documented')

            # Rileva pattern di test
            if method_name.startswith('Test'):
                attributes.append('test')
            elif method_name.startswith('Benchmark'):
                attributes.append('benchmark')
            elif method_name.startswith('Example'):
                attributes.append('example')

            # Go non ha modificatori di accesso espliciti
            # La visibilità è determinata dalla prima lettera del nome
            access_modifier = 'public' if method_name[0].isupper() else 'private'

            return MethodInfo(
                name=method_name,
                body=method_body.strip(),
                attributes=attributes,
                access_modifier=access_modifier,
                line_start=line_start,
                line_end=line_end,
                language=Language.GO,
                parameters=parameters,
                return_type=return_type.strip() if return_type else ""
            )
        except Exception as e:
            print(f"Errore nel parsing del metodo Go: {e}")
            return None

    def get_mock_filters(self) -> List[str]:
        return [
            'gomock.NewController(',
            'NewMock',
            '.EXPECT(',
            '.Return(',
            'testify/mock.Mock',
            '.On(',
            '.AssertCalled(',
            '.AssertExpectationsMet(',
        ]
