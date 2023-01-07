from pathlib import Path
import simulation
import marker_code
import evaluation
import numpy as np
import json
import yaml
import os
import shutil

def main():
    
    # Load configurations
    with open('./config/config-sim.yaml') as f:
        cfg = yaml.safe_load(f)
        
    # Paths for output
    path_output = Path(cfg['output_path']) if os.path.isabs(cfg['output_path']) \
        else Path().resolve() / cfg['output_path']
    global path_experiments
    path_experiments = path_output / 'expriments'
    
    # Experiment configurations
    ids_rates = cfg['IDS_rates']
    seq_cfgs = cfg['sequence_configurations']
    coverages = cfg['coverages']
    random_seed = cfg['random_seed']
    cluster_num = cfg['cluster_number']
    
    for rate in ids_rates:
            for seqcfg in seq_cfgs:
                for cv in coverages:
                    run_simulation(
                        ins_p = rate['ins_p'],
                        del_p = rate['del_p'],
                        sub_p = rate['sub_p'],
                        marker_num      = seqcfg['marker_number'],
                        marker_len      = seqcfg['marker_length'],
                        cluster_num     = cluster_num,
                        cluster_size    = cv,
                        data_len        = seqcfg['data_length'],
                        random_seed     = random_seed
                    )
                    
    return
    

def run_simulation(ins_p, del_p, sub_p, marker_num, marker_len, cluster_num, cluster_size, data_len, random_seed):

    cfgstr = 'i' + str(int(ins_p*100)) + 'd' + str(int(del_p*100)) + 's' + str(int(sub_p*100)) + \
        '-dlen' + str(data_len) + '-mk' + str(marker_len) + '*' + str(marker_num) + '-cv' + str(cluster_size)
        
    path_exp = path_experiments / cfgstr
    _path_byprod = path_exp / 'byproduct'
    
    _path_encode    = _path_byprod / 'encode'
    path_sequence   = _path_encode / 'input/sequences.txt'
    path_marker     = _path_encode / 'input/markers.txt'
    path_encoded    = _path_encode / 'output/encoded_sequences.txt'
    path_mkconfig     = _path_encode / 'output/marker_config.json'
    
    _path_decode    = _path_byprod / 'decode'
    path_cluster    = _path_decode / 'input/clusters.txt'
    path_decoded    = _path_decode / 'output/decoded.txt'
    path_decoded_with_mark = _path_decode / 'output/decoded_with_marker.txt'
    
    path_evaluation = path_exp
    path_exp_config = path_exp / 'configuration.json'
    
    path_done = _path_byprod / '.done'
    
    cfg = {
        'ins_p': ins_p,
        'del_p': del_p,
        'sub_p': sub_p,
        'marker_num': marker_num,
        'marker_len': marker_len,
        'coverage': cluster_size,
        'sequence_len': data_len + marker_num * marker_len,
        'data_len': data_len,
        'random_seed': random_seed,
        'cluster_num': cluster_num
    }
    
    if path_done.exists():
        return
    elif path_exp.exists():
        shutil.rmtree(path_exp)
        
    _path_encode.mkdir(exist_ok=True, parents=True)
    _path_decode.mkdir(exist_ok=True, parents=True)
    path_evaluation.mkdir(exist_ok=True, parents=True)
        
    with path_exp_config.open('w') as f:
        json.dump(cfg, f, indent=2)
               
    SYMBOLS = ['A', 'C', 'G', 'T']
    cluster_seperator = ('=' * 20)

    # Initialize simulation
    sim = simulation.Simulation(ins_p=ins_p, del_p=del_p, sub_p=sub_p, symbols=SYMBOLS, seed=random_seed)

    # Generate simulation data
    sim.generate_simulation_data(
        seq_num        = cluster_num,
        data_len       = data_len,
        marker_num     = marker_num,
        marker_len     = marker_len,
        sequence_path  = str(path_sequence),
        marker_path    = str(path_marker),
    )

    # Encode
    encoder = marker_code.Encoder()
    encoder.encode(
        sequence_path   = str(path_sequence),
        marker_path     = str(path_marker),
        encoded_path    = str(path_encoded),
        config_path     = str(path_mkconfig)
    )

    # Simulate a IDS channel
    sim.simulate_IDS_channel(
        encoded_path    = str(path_encoded),
        output_path     = str(path_cluster),
        sample_num      = cluster_size,
        seperator       = cluster_seperator
    )

    # Decode
    decoder = marker_code.Decoder(ins_p=ins_p, del_p=del_p, sub_p=sub_p, symbols=SYMBOLS, np_dtype=np.float64)
    decoder.decode(
        cluster_path             = str(path_cluster),
        config_path              = str(path_mkconfig),
        decoded_path             = str(path_decoded),
        decoded_with_marker_path = str(path_decoded_with_mark),
        cluster_seperator        = cluster_seperator,
    )

    # Evaluate with markers
    evaluation.report(
        truth_path      = str(path_encoded),
        result_path     = str(path_decoded_with_mark),
        output_dir      = str(path_evaluation),
    )
    
    with path_done.open('w') as f:
        f.write("1")
    
    return


if __name__ == '__main__':
    main()