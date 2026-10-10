from lob.terminals.registry import TERMINALS, TerminalProfile, get_terminal, register_terminal
from lob.terminals import generic as _generic, metascalp as _metascalp  # noqa: F401  (регистрация)

__all__ = ["TERMINALS", "TerminalProfile", "get_terminal", "register_terminal"]
