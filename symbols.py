import copy

def symbol_init(symbols):
    global SYMBOLS, MAP, REV_MAP
    SYMBOLS = list(symbols)
    MAP = {s: i for i, s in enumerate(SYMBOLS)}
    REV_MAP = {v: k for k, v in MAP.items()}

def symbol2digit(sequence: str):
    return [MAP[c] for c in str]

def digit2symbol(digits: list(int)):
    return ''.join(REV_MAP[digit] for digit in digits)

def all_symbols():
    return copy.deepcopy(SYMBOLS)