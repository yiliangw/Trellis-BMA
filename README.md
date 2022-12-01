# Trace Reconstruction with Marker Code

This is a python implementation for trace reconstruction with marker code [[1]](#r1) in the context of DNA-based data storage. You can run it directly with either the [CNR dataset](https://github.com/microsoft/clustered-nanopore-reads-dataset) [[2]](#r2) or a simple simulated IDS channel ([IDS_channel.py](IDS_channel.py)). A [derivation](derivation/derivation.pdf) for relevent formulas is also provided.

## Quick Start
```
pip3 install -r requirements.txt
python3 main.py
```
This will process all of the 10,000 clusters in the CNR dataset and may take some time. If you want to run with fewer clusters, please configure `CLUSTER_NUM` in the `main()` function of [main.py](main.py). The results and statistics will be available in `output/` after the program terminates.

## Dependencies

+ python3 (tested on python3.8)
+ numpy
+ matplotlib
+ tqdm


## Configurations

For available configurations, please check the `main()` function of [main.py](main.py).

## Derivation

A [derivation](derivation/derivation.pdf) for relevent formulas is available in `derivation/`, with LaTeX [source](derivation/derivation.tex). 

## References

<a id="r1">[1]</a> E. A. Ratzer, “Marker codes for channels with insertions and deletions,” Ann. Télécommun., vol. 60, no. 1–2, pp. 29–44, Feb. 2005, doi: 10.1007/BF03219806.

<a id="r2">[2]</a> S. R. Srinivasavaradhan, M. Du, S. Diggavi, and C. Fragouli, “Symbolwise MAP for Multiple Deletion Channels,” in 2019 IEEE International Symposium on Information Theory (ISIT), Paris, France, Jul. 2019, pp. 181–185. doi: 10.1109/ISIT.2019.8849567.

<a id="r3">[3]</a> C. M. Bishop and N. M. Nasrabadi, Pattern recognition and machine learning, vol. 4. Springer, 2006.

<a id="r4">[4]</a> R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison, Biological sequence analysis: probabilistic models of proteins and nucleic acids. Cambridge university press, 1998.

<a id="r5">[5]</a> D. Lin, Y. Tabatabaee, Y. Pote, and D. Jevdjic, “Managing reliability skew in DNA storage,” in Proceedings of the 49th Annual International Symposium on Computer Architecture, New York New York, Jun. 2022, pp. 482–494. doi: 10.1145/3470496.3527441.

<a id="r6">[6]</a> M. C. Davey and D. J. C. Mackay, “Reliable communication over channels with insertions, deletions, and substitutions,” IEEE Trans. Inform. Theory, vol. 47, no. 2, pp. 687–698, Feb. 2001, doi: 10.1109/18.910582.

<a id="r7">[7]</a> S. R. Srinivasavaradhan, S. Gopi, H. D. Pfister, and S. Yekhanin, “Trellis BMA: Coded trace reconstruction on IDS channels for DNA storage,” in 2021 IEEE International Symposium on Information Theory (ISIT), 2021, pp. 2453–2458.
