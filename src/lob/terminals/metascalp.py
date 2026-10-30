from lob.terminals.registry import TerminalProfile, register_terminal

METASCALP = register_terminal(TerminalProfile(
    name="metascalp", layout_kind="ladder",
    calibration={"ink_threshold": 70},
    notes=("Множитель шага цены 1, сжатие выключено; автопрокрутка включена; "
           "полосы объёма и подсветка крупных объёмов приглушены; единицы объёма не менять "
           "в ходе записи; запись в 100% масштабе, постоянная частота кадров.")))