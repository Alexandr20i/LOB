from lob.terminals.registry import TERMINALS, TerminalProfile, get_terminal, register_terminal

__all__ = ["TERMINALS", "get_terminal", "register_terminal", "TerminalProfile"]

# Импорт всех терминалов для регистрации. generic нужен: на него ссылается
# значение по умолчанию в scripts/calibrate.py.
from lob.terminals import generic, metascalp  # noqa: E402, F401