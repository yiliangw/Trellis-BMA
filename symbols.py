import copy

def init(symbols=['A', 'C', 'G', 'T']):
    global SYMBOLS, MAP, REV_MAP
    SYMBOLS = list(symbols)
    MAP = {s: i for i, s in enumerate(SYMBOLS)}
    REV_MAP = {v: k for k, v in MAP.items()}

def symbol2digit(sequence: str):
    return [MAP[c] for c in sequence]

def digit2symbol(digits: list):
    return ''.join(REV_MAP[digit] for digit in digits)

def all():
    return copy.deepcopy(SYMBOLS)

def num():
    return len(SYMBOLS)

# Initialize symbols to be DNA nucleotides as default
init()