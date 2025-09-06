from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class Language(Enum):
    CSHARP = "csharp"
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    PHP = "php"
    RUBY = "ruby"
    GO = "go"


@dataclass
class MethodInfo:
    name: str
    body: str
    attributes: List[str]
    access_modifier: str
    line_start: int
    line_end: int
    language: Language
    parameters: str = ""
    return_type: str = ""


class LanguageParser(ABC):
    """Classe base astratta per i parser specifici di ogni linguaggio"""

    @abstractmethod
    def get_method_pattern(self) -> str:
        """Restituisce il pattern regex per identificare i metodi"""
        pass

    @abstractmethod
    def get_test_attributes(self) -> List[str]:
        """Restituisce i pattern per identificare i metodi di test"""
        pass

    @abstractmethod
    def parse_method_match(self, match, content: str, lines: List[str]) -> Optional[MethodInfo]:
        """Analizza una corrispondenza regex per creare MethodInfo"""
        pass

    @abstractmethod
    def get_mock_filters(self) -> List[str]:
        """Restituisce i pattern per identificare i mock nei test"""
        pass

    @abstractmethod
    def get_comments_config(self) -> Dict[str, Any]:
        """Restituisce i pattern per commentare i test non selezionati"""
        pass