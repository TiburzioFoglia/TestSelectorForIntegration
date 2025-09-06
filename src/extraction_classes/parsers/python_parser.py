import re
import ast
from typing import List, Optional, Dict, Any
from src.extraction_classes.base_classes import LanguageParser, MethodInfo, Language


class MethodVisitor(ast.NodeVisitor):
    """
    Visitatore AST che attraversa l'albero sintattico e raccoglie
    informazioni su tutte le definizioni di funzione (metodi e funzioni).
    """

    def __init__(self, source_code: str):
        # Usare splitlines() è più sicuro
        self.source_lines = source_code.splitlines()
        self.methods: List[MethodInfo] = []

    def get_source_segment(self, node: ast.AST) -> str:
        """
        Funzione helper robusta per estrarre il segmento di codice sorgente di un nodo.
        Gestisce i casi limite in cui le linee o le colonne non sono disponibili.
        """
        if not (hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and
                hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset')):
            return ""  # Non è possibile estrarre se mancano le informazioni

        start_line, end_line = node.lineno - 1, node.end_lineno - 1

        # Gestione dei casi limite per le linee
        if start_line < 0 or end_line >= len(self.source_lines):
            # Prova a recuperare il testo parziale se possibile
            start_line = max(0, start_line)
            end_line = min(len(self.source_lines) - 1, end_line)

        if start_line > end_line:
            return ""  # Caso impossibile

        # Se il nodo è su una singola linea
        if start_line == end_line:
            return self.source_lines[start_line][node.col_offset:node.end_col_offset]

        # Se il nodo si estende su più linee
        first_line = self.source_lines[start_line][node.col_offset:]
        middle_lines = self.source_lines[start_line + 1:end_line]
        last_line = self.source_lines[end_line][:node.end_col_offset]

        return "\n".join([first_line] + middle_lines + [last_line])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Questo metodo viene chiamato per ogni 'def' nel codice."""

        # Estrazione del corpo completo del metodo, inclusi decoratori e firma
        start_line_num = node.lineno
        end_line_num = node.end_lineno

        # I decoratori possono iniziare prima della linea della funzione
        if node.decorator_list:
            start_line_num = node.decorator_list[0].lineno

        # Aggiustamento per l'indice basato su 0
        start_idx = start_line_num - 1
        end_idx = end_line_num

        # Assicurarsi che gli indici siano nei limiti
        if start_idx < 0: start_idx = 0
        if end_idx > len(self.source_lines): end_idx = len(self.source_lines)

        # Usiamo le linee originali per preservare la formattazione
        method_body_lines = self.source_lines[start_idx:end_idx]
        method_body = "\n".join(method_body_lines)

        # Estrae gli attributi (decoratori) usando la nostra funzione helper
        attributes = [f"@{self.get_source_segment(d)}" for d in node.decorator_list]

        # Estrae i parametri
        params_list = []
        # Argomenti posizionali e con keyword
        for arg in node.args.args:
            params_list.append(arg.arg)
        # *args
        if node.args.vararg:
            params_list.append(f"*{node.args.vararg.arg}")
        # Argomenti solo keyword
        for arg in node.args.kwonlyargs:
            params_list.append(arg.arg)
        # **kwargs
        if node.args.kwarg:
            params_list.append(f"**{node.args.kwarg.arg}")
        parameters = ", ".join(params_list)

        # Estrae il tipo di ritorno
        return_type = self.get_source_segment(node.returns) if node.returns else ""

        # Determina l'access modifier per convenzione
        access_modifier = 'public'
        if node.name.startswith('__') and not node.name.endswith('__'):
            access_modifier = 'private'
        elif node.name.startswith('_'):
            access_modifier = 'protected'

        method_info = MethodInfo(
            name=node.name,
            body=method_body.strip(),
            attributes=attributes,
            access_modifier=access_modifier,
            line_start=start_line_num,
            line_end=end_line_num,
            language=Language.PYTHON,
            parameters=parameters,
            return_type=return_type
        )
        self.methods.append(method_info)

        # Per evitare di visitare funzioni annidate all'interno di questa funzione.
        # Se si volessero anche le funzioni interne, rimuovere il return.
        return


class PythonParser(LanguageParser):
    """Parser specifico per il linguaggio Python basato su AST."""

    def get_method_pattern(self) -> str:
        return ""  # Non usato

    def get_test_attributes(self) -> List[str]:
        return [
            'pytest',
            'unittest',
            'def test_'
        ]

    def extract_methods_with_ast(self, content: str) -> List[MethodInfo]:
        """Estrae i metodi usando il modulo AST per un parsing robusto."""
        if not content.strip():
            return []

        try:
            # Aggiunge un newline finale se manca, per la robustezza di ast.parse
            if not content.endswith('\n'):
                content += '\n'
            tree = ast.parse(content)
            visitor = MethodVisitor(content)
            visitor.visit(tree)
            return visitor.methods
        except SyntaxError as e:
            print(f"Errore di sintassi nel file Python, impossibile usare AST: {e}")
            return []
        except Exception as e:
            print(f"Errore imprevisto durante il parsing AST: {e}")
            return []

    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        return None  # Non usato

    def get_mock_filters(self) -> List[str]:
        # \b -> word boundary (confine di parola), per evitare match parziali come "notmock.patch"
        # \s* -> zero o più spazi, per tollerare la formattazione
        return [
            # Utilizzo di @patch come decoratore
            r'@\s*patch\b',
            r'@\s*mock\.patch\b',

            # Chiamate a patch e Mock
            r'\bpatch\s*\(',
            r'\bmock\.patch\s*\(',
            r'\bMock\s*\(',
            r'unittest\.mock\.patch\b',
            r'unittest\.mock\.Mock\b',

            # Metodi comuni sugli oggetti mock
            r'\.assert_called_with\b',
            r'\.assert_called_once_with\b',
            r'\.assert_any_call\b',
            r'\.assert_has_calls\b',
            r'\.assert_not_called\b',

            # Configurazione del comportamento del mock
            r'\.return_value\b',
            r'\.side_effect\b',
            r'\.spec\b',

            # Contesto 'with' per il patching
            r'with\s+patch\b',
            r'with\s+mock\.patch\b',
        ]

    def get_comments_config(self) -> Dict[str, Any]:
        return {
            'comment_symbol': '#',
            'method_regex': re.compile(r'^\s*def\s+(?P<method_name>[a-zA-Z_0-9]+)\s*\(')
        }