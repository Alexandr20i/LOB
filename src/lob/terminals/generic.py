from lob.terminals.registry import TerminalProfile, register_terminal

GENERIC = register_terminal(TerminalProfile(
    name="generic", layout_kind="ladder",
    notes="Любой терминал с одной лестницей цен (размер и цена в строках одинаковой высоты)."))
