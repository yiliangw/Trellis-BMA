from pathlib import Path
import simulation
import marker_code
import evaluation
import numpy as np
import json

def main():
    
    ############################
    ###### Configurations ######
    ############################
    CLUSTER_NUM     = 1000        # The number of clusters to process
    RANDOM_SEED     = 6219      # Random seed for IDS channel
    ############################
    ############################
    
    total_len = 110
    err_rates = [0.01, 0.03, 0.05]
    marker_cfg = [
        (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11), (1,12),
        (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
        (3,1), (3,2), (3,3), (3,4),
        (4,1), (4,2), (4,3),
        (6,1), (6,2)
    ]
    cluster_sizes = [4, 6, 8, 10]
    
    # total_len = [80, 120, 160, 200]
    # err_rates = [0.01, 0.03, 0.05]
    # cluster_sizes = [6]
    
    for mklen, mknum in marker_cfg:
        for er in err_rates:
            for size in cluster_sizes:
            
                seqlen = total_len - (mknum * mklen)
                        
                run_simulation(
                    ins_p = er,
                    del_p = er,
                    sub_p = er,
                    marker_num      = mknum,
                    marker_len      = mklen,
                    cluster_num     = CLUSTER_NUM,
                    cluster_size    = size,
                    seq_len         = seqlen,
                    random_seed     = RANDOM_SEED
                )
                
    return
    

def run_simulation(ins_p, del_p, sub_p, marker_num, marker_len, cluster_num, cluster_size, seq_len, random_seed):

    cfgstr =  'data/sim/' 'i' + str(int(ins_p*100)) + 'd' + str(int(del_p*100)) + 's' + str(int(sub_p*100)) + \
        '-len' + str(seq_len) + '-mk' + str(marker_len) + '*' + str(marker_num) + '-cv' + str(cluster_size)
    
    cfg = {
        'ins_p': ins_p,
        'del_p': del_p,
        'sub_p': sub_p,
        'marker_num': marker_num,
        'marker_len': marker_len,
        'coverage': cluster_size,
        'sequence_length': seq_len,
    }
    
    data_path = Path(__file__).resolve().parent / cfgstr
    evaluation_path = data_path / 'evaluation'
    if evaluation_path.exists():
        return
    data_path.mkdir(parents=True, exist_ok=True)
    
    fname_cfg = 'configuration.json'
    
    with (data_path / fname_cfg).open('w') as f:
        json.dump(cfg, f, indent=4)
               
    encode_path = data_path / 'encode'
    
    sequence_path_str   = str(encode_path / 'input/sequences.txt')
    marker_path_str     = str(encode_path / 'input/markers.txt')
    encoded_path_str    = str(encode_path / 'output/encoded_sequences.txt')
    config_path_str     = str(encode_path / 'output/marker_config.json')

    decode_path = data_path / 'decode'
    cluster_path_str    = str(decode_path / 'input/clusters.txt')
    decoded_path_str    = str(decode_path / 'output/decoded.txt')
    decoded_with_mark_path_str \
                        = str(decode_path / 'output/decoded_with_marker.txt')

    evaluation_path_str = str(evaluation_path)

    SYMBOLS = ['A', 'C', 'G', 'T']
    cluster_seperator = ('=' * 20)

    # Initialize simulation
    sim = simulation.Simulation(ins_p=ins_p, del_p=del_p, sub_p=sub_p, symbols=SYMBOLS, seed=random_seed)

    # Generate simulation data
    sim.generate_simulation_data(
        seq_num        = cluster_num,
        seq_len        = seq_len,
        marker_num     = marker_num,
        marker_len     = marker_len,
        sequence_path  = sequence_path_str,
        marker_path    = marker_path_str,
    )

    # Encode
    encoder = marker_code.Encoder()
    encoder.encode(
        sequence_path   = sequence_path_str,
        marker_path     = marker_path_str,
        encoded_path    = encoded_path_str,
        config_path     = config_path_str
    )

    # Simulate a IDS channel
    sim.simulate_IDS_channel(
        encoded_path    = encoded_path_str,
        output_path     = cluster_path_str,
        sample_num      = cluster_size,
        seperator       = cluster_seperator
    )

    # Decode
    decoder = marker_code.Decoder(ins_p=ins_p, del_p=del_p, sub_p=sub_p, symbols=SYMBOLS, np_dtype=np.float64)
    decoder.decode(
        cluster_path             = cluster_path_str,
        config_path              = config_path_str,
        decoded_path             = decoded_path_str,
        decoded_with_marker_path = decoded_with_mark_path_str,
        cluster_seperator        = cluster_seperator,
    )

    # Evaluate with markers
    evaluation.report(
        truth_path      = encoded_path_str,
        result_path     = decoded_with_mark_path_str,
        output_dir      = evaluation_path_str,
    )
    
    return


if __name__ == '__main__':
    main()