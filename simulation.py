import random
import pathlib
import copy


class Simulation():

    def __init__(self, ins_p: float, del_p: float, sub_p: float, symbols=['A', 'C', 'G', 'T'], seed=0):
        self.symbols = copy.deepcopy(symbols)
        self.ins_p = ins_p
        self.del_p = del_p
        self.sub_p = sub_p
        self.seed = seed

    def generate_simulation_data(self, seq_num, seq_len, marker_num, marker_len, sequence_path: str, marker_path: str):
    
        fseq = pathlib.Path(sequence_path)
        fmarker = pathlib.Path(marker_path)

        fseq.parent.mkdir(parents=True, exist_ok=True)
        fmarker.parent.mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)
        symbols = self.symbols
        with fseq.open('w') as f:
            for _ in range(seq_num):
                f.write(''.join(random.choices(symbols, k=seq_len)) + '\n')

        with fmarker.open('w') as f:
            for i in range(1, marker_num+1):
                position = int(seq_len/(marker_num+1)*i)
                marker = ''.join(random.choices(symbols, k=marker_len))
                f.write('{} {}\n'.format(position, marker))
    

    def simulate_IDS_channel(self, encoded_path: str, output_path: str, sample_num=5, seperator=('='*20)):

        pencoded = pathlib.Path(encoded_path)
        poutput = pathlib.Path(output_path)

        poutput.parent.mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)

        with pencoded.open('r') as fencoded, poutput.open('w') as foutput:
            for encoded in fencoded:
                encoded = encoded.strip()
                samples = self.__generate_noisy_samples(encoded, sample_num)
                foutput.writelines('\n'.join(samples) + '\n' + seperator + '\n')
        

    def __generate_noisy_samples(self, gold, sample_num):
        symbols = self.symbols
        sub_dict = {s: list(set(symbols)-set([s])) for s in symbols}
        res = []

        for _ in range(sample_num):
            sample = []
            i = 0
            while i < len(gold):
                r = random.random()
                if r < self.ins_p:  # Insertion
                    sample.append(random.choice(symbols))
                elif r < self.ins_p + self.del_p:   # Deletion
                    i += 1
                elif r < self.ins_p + self.del_p + self.sub_p:  # Substitution
                    sample.append(random.choice(sub_dict[gold[i]]))
                    i += 1
                else:   # No error
                    sample.append(gold[i])
                    i += 1
            res.append(''.join(sample))

        return res
    