from pathlib import Path
import evaluation
import marker_code
import json


def main():

    path_cnr = Path().resolve() / 'dataset'
    path_clusters = path_cnr / 'Clusters.txt'
    global path_centers
    path_centers = path_cnr / 'Centers.txt'
    
    path_output = Path().resolve() / 'data/CNR'

    global path_fmt_clusters, path_experiments
    path_fmt_clusters = path_output / 'format_clusters.txt'
    path_experiments = path_output / 'experiments'
    
    ids_rates = [0.08, 0.01, 0.015, 0.02, 0.025]
    marker_nums = [1, 6, 10]
    marker_lengths = [1]
    coverages = [4, 6]

    # Format the cluster file to make it compatible to the marker_code module
    format_CNR_cluster(path_clusters, path_fmt_clusters)

    for ids_rate in ids_rates:
        for marker_num in marker_nums:
            for coverage in coverages:
                run_CNR(
                    ins_p        = ids_rate,
                    del_p        = ids_rate,
                    sub_p        = ids_rate,
                    marker_num   = marker_num,
                    marker_len   = 1,
                    coverage     = coverage
                )


def format_CNR_cluster(path_clusters: Path, path_fmt_clusters: Path):
    path_fmt_clusters.parent.mkdir(exist_ok=True, parents=True)
    with path_clusters.open('r') as fclu, path_fmt_clusters.open('w') as ffmat:
        seperator = fclu.readline() # Skip the first line
        for line in fclu:
            ffmat.write(line)
        ffmat.write(seperator)  # Append a boundary to the formatted file
    return

def run_CNR(ins_p, del_p, sub_p, marker_num, marker_len, coverage):

    cfgstr =  'i' + str(round(ins_p, 3)) + 'd' + str(round(del_p, 3)) + 's' + str(round(sub_p, 3)) + '-mk' + str(marker_len) + '*' + str(marker_num) + '-cv' + str(coverage)

    path_exp = path_experiments / cfgstr
    path_mkcfg = path_exp / 'mk-config.json'
    path_cfg = path_exp / 'configuration.json'
    path_decoded = path_exp / 'decoded.txt'
    path_decoded_with_marker = path_exp / 'decoded_with_marker.txt'
    path_evaluation = path_exp / 'evaluation'

    if path_evaluation.exists():
        return

    path_exp.mkdir(exist_ok=True, parents=True)

    # Split the markers according to the configuration of the experiment
    spliter = marker_code.DatasetSpliter()
    spliter.split_dataset(
        marker_len  = 1,
        marker_num  = marker_num,
        sequence_path  = str(path_centers),
        config_path    = str(path_mkcfg),
    )

    # Save the configuration of the experiment
    cfg = {
        'ins_p': ins_p,
        'del_p': del_p,
        'sub_p': sub_p,
        'marker_num': marker_num,
        'marker_len': marker_len,
        'coverage': coverage,
    }
    with path_cfg.open('w') as f:
        json.dump(cfg, f)

    decoder = marker_code.Decoder(ins_p=ins_p, del_p=del_p, sub_p=sub_p)

    decoder.decode(
        cluster_path    = str(path_fmt_clusters),
        config_path     = str(path_mkcfg),
        decoded_path    = str(path_decoded),
        decoded_with_marker_path = str(path_decoded_with_marker)
    )

    evaluation.report(
        truth_path      = str(path_centers),
        result_path     = str(path_decoded_with_marker),
        output_dir      = str(path_evaluation)
    )
        

if __name__ == '__main__':
    main()
