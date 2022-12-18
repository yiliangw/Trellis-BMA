import random
import json
import pathlib
import symbols as sym





def generate_simulation_data(seq_num, seq_len, marker_num, marker_len, sequence_path: str, marker_path: str, seed: int):
   
    fseq = pathlib.Path(sequence_path)
    fmarker = pathlib.Path(marker_path)

    fseq.parent.mkdir(parents=True, exist_ok=True)
    fmarker.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    symbols = sym.all()
    with fseq.open('w') as f:
        for _ in range(seq_num):
            f.write(''.join(random.choices(symbols, k=seq_len)) + '\n')

    with fmarker.open('w') as f:
        for i in range(1, marker_num+1):
            position = int(seq_len/(marker_num+1)*i)
            marker = ''.join(random.choices(symbols, k=marker_len))
            f.write('{} {}\n'.format(position, marker))
    

def simulate_IDS_channel(encoded_path: str, output_path: str, ins_p: float, del_p: float, sub_p: float, 
    sample_num=5, seed=0, seperator=('='*20)):

    pencoded = pathlib.Path(encoded_path)
    poutput = pathlib.Path(output_path)

    poutput.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    with pencoded.open('r') as fencoded, poutput.open('w') as foutput:
        for encoded in fencoded:
            encoded = encoded.strip()
            samples = generate_noisy_samples(encoded, sample_num, ins_p, del_p, sub_p)
            foutput.writelines('\n'.join(samples) + '\n' + seperator + '\n')
    

def generate_noisy_samples(gold, sample_num, sub_p, del_p, ins_p):
    symbols = sym.all()
    sub_dict = {s: list(set(symbols)-set([s])) for s in symbols}
    res = []

    for _ in range(sample_num):
        sample = []
        i = 0
        while i < len(gold):
            r = random.random()
            if r < ins_p:
                sample.append(random.choice(symbols))
            elif r < ins_p + del_p:
                i += 1
            elif r < ins_p + del_p + sub_p:
                sample.append(random.choice(sub_dict[gold[i]]))
                i += 1
            else:
                sample.append(gold[i])
                i += 1
        res.append(''.join(sample))

    return res
    