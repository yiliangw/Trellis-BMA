from pathlib import Path
import os
import evaluation
import marker_code
import json
import yaml
import random
import shutil

def main():

    # Load configurations
    with open('./config/config-cnr.yaml') as f:
        cfg = yaml.safe_load(f)
    
    # Paths for CRN dataset
    path_cnr = Path(cfg['CNR_path']) if os.path.isabs(cfg['CNR_path']) \
        else Path().resolve() / cfg['CNR_path']
    global path_centers, path_clusters
    path_centers = path_cnr / 'Centers.txt'
    path_clusters = path_cnr / 'Clusters.txt'
    
    # Paths for output
    path_output = Path(cfg['output_path']) if os.path.isabs(cfg['output_path']) \
        else Path().resolve() / cfg['output_path']
    global path_experiments
    path_experiments = path_output / 'experiments'

    # Experiment configurations
    ids_rates = cfg['IDS_rates']
    marker_cfgs = cfg['marker_configurations']
    coverages = cfg['coverages']
    cluster_num = cfg['cluster_number']
    if cluster_num != None and cluster_num <= 0:
        cluster_num = None
    random_seed = cfg['random_seed']

    for rate in ids_rates:
        for mkcfg in marker_cfgs:
            for cv in coverages:
                if cv != None and cv <= 0:
                    cv = None
                run_CNR(
                    ins_p   = rate['ins_p'],
                    del_p   = rate['del_p'],
                    sub_p   = rate['sub_p'],
                    marker_num  = mkcfg['number'],
                    marker_len  = mkcfg['length'],
                    coverage    = cv,
                    cluster_num = cluster_num,
                    random_seed = random_seed
                )


def run_CNR(ins_p, del_p, sub_p, marker_num, marker_len, coverage=None, cluster_num=None, random_seed=6219):

    cfgstr =  'i' + str(round(ins_p, 3)) + 'd' + str(round(del_p, 3)) + 's' + str(round(sub_p, 3)) + '-mk' + str(marker_len) + '*' + str(marker_num) + '-cv' + str(coverage) + '-nclus' + str(cluster_num)

    path_exp = path_experiments / cfgstr
    path_byprod = path_exp / 'byproduct'
    path_exp_centers = path_byprod / 'centers.txt'
    path_exp_clusters = path_byprod / 'clusters.txt'
    path_mkcfg = path_byprod / 'mk-config.json'
    path_cfg = path_exp / 'configuration.json'
    path_decoded = path_byprod / 'decoded.txt'
    path_decoded_with_marker = path_byprod / 'decoded_with_marker.txt'
    path_evaluation = path_exp
    path_done = path_byprod / '.done'

    if path_done.exists():
        return
    elif path_exp.exists():
        shutil.rmtree(path_exp)

    path_exp.mkdir(exist_ok=True, parents=True)
    path_byprod.mkdir(exist_ok=True, parents=True)
    path_evaluation.mkdir(exist_ok=True, parents=True)

    # Format the CNR cluster file to make it compatible to the marker_code module
    format_CNR_cluster(
        path_centers      = path_centers,
        path_clusters     = path_clusters,
        path_fmt_centers  = path_exp_centers,
        path_fmt_clusters = path_exp_clusters,
        coverage          = coverage,
        cluster_num       = cluster_num,
        random_seed       = random_seed
    )

    # Split the markers according to the configuration of the experiment
    spliter = marker_code.DatasetSpliter()
    spliter.split_dataset(
        marker_len  = marker_len,
        marker_num  = marker_num,
        sequence_path  = str(path_exp_centers),
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
        'cluster_num': cluster_num,
        'random_seed': random_seed
    }
    with path_cfg.open('w') as f:
        json.dump(cfg, f, indent=2)

    decoder = marker_code.Decoder(ins_p=ins_p, del_p=del_p, sub_p=sub_p)

    decoder.decode(
        cluster_path    = str(path_exp_clusters),
        config_path     = str(path_mkcfg),
        decoded_path    = str(path_decoded),
        decoded_with_marker_path = str(path_decoded_with_marker)
    )

    evaluation.report(
        truth_path      = str(path_exp_centers),
        result_path     = str(path_decoded_with_marker),
        output_dir      = str(path_evaluation)
    )
    
    with path_done.open('w') as f:
        f.write('1')
    
    return
        

def format_CNR_cluster(path_clusters: Path, path_fmt_clusters: Path, path_centers: Path, path_fmt_centers: Path, coverage=None, cluster_num=None, random_seed=6219):

    if random_seed != None:
        random.seed(random_seed)

    path_fmt_clusters.parent.mkdir(exist_ok=True, parents=True)
    path_fmt_centers.parent.mkdir(exist_ok=True, parents=True)

    with path_clusters.open('r') as f, path_fmt_clusters.open('w') as ffmat:
        seperator = f.readline() # Skip the first line
        cluster_cnt = 0
        flush_cnt = 0       # flush_cnt does not always equal to cluster_cnt
                            # because the last line of Clusters.txt is not a
                            # seperator. flush_cnt is the real number of clusters,
                            # which may be smaller than the parameter cluster_num.
        cluster = []
        zero_clusters = []
        def flush_cluster():
            nonlocal cluster, flush_cnt
            if coverage == None:
                clu = cluster
            elif len(cluster) == 0:
                clu = []
                zero_clusters.append(flush_cnt)
            elif coverage > len(cluster):
                clu = cluster * int(coverage / len(cluster))
                clu += random.sample(cluster, coverage-len(clu))
            else:
                clu = random.sample(cluster, coverage)
            
            if not len(clu) == 0:
                ffmat.writelines(clu)
                ffmat.write(seperator)
            flush_cnt += 1
            cluster = []

        for line in f:
            if line.startswith(seperator[0]):
                cluster_cnt += 1
                if cluster_num != None and cluster_cnt == cluster_num:
                    break
                else:
                    flush_cluster()
            else:
                cluster.append(line)

        flush_cluster() # Flush the last cluster

    with path_centers.open('r') as f, path_fmt_centers.open('w') as ffmat:
        for i in range(flush_cnt):
            line = f.readline()
            if i not in zero_clusters:
                ffmat.write(line)

    return


if __name__ == '__main__':
    main()
