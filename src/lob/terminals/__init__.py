from lob.terminals.registry import TERMINALS, get_terminal, register_terminal, TerminalProfile

__all__ = ["TERMINALS", "get_terminal", "register_terminal", "TerminalProfile"]

# Импорт всех терминалов для регистрации
from lob.terminals import metascalp  # noqa: F401