import random
import json
import pathlib
import symbols


def __gen_noisy_sample(gold, sub_p, del_p, ins_p):
    symbols = sym.all()
    res = []
    for w in gold:
        r = random.random()
        if r < sub_p:
            res.append(random.choice(symbols))
        elif r < sub_p + ins_p:
            res.append(random.choice(symbols))
            res.append(w)
        elif r > sub_p+ins_p+del_p:
            res.append(w)
    return ''.join(res)

def generate_noisy_samples(gold, n, sub_p, del_p, ins_p):
    res = []
    for i in range(n):
        res.append(__gen_noisy_sample(gold, sub_p, del_p, ins_p))
    return res


def generate_encode_input(seq_num, seq_len, marker_num, marker_len, sequence_path: str, marker_path: str, seed):
   
    fseq = pathlib.Path(sequence_path)
    fmarker = pathlib.Path(marker_path)

    fseq.parent.mkdir(parents=True, exist_ok=True)
    fmarker.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    with fseq.open('w') as f:
        for _ in range(seq_num):
            f.write(''.join(random.choices(symbols.all(), k=seq_len)) + '\n')

    with fmarker.open('w') as f:
        for i in range(1, marker_num+1):
            position = int(seq_len/(marker_num+1)*i)
            marker = ''.join(random.choices(symbols.all(), k=marker_len))
            f.write('{} {}\n'.format(position, marker))
    