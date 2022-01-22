# Temporal Knowledge Graph Completion using a Linear Temporal Regularizer and Multivector Embeddings


## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name telm_env python=3.7
source activate telm_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets


Once the datasets are downloaded, go to the tkbc/ folder and add them to the package data folder by running :
```
python process_icews.py
python process_timegran.py --tr 100 --dataset yago11k
python process_timegran.py --tr 1 --dataset wikidata12k
# For wikidata11k and yago12k, change the tr for changing time granularity
```

This will create the files required to compute the filtered metrics.

## Reproducing results of TeLM

In order to reproduce the results of TeLM on the four datasets in the paper, go to the tkbc/ folder and run the following commands

```
python learner.py --dataset ICEWS14 --model TeLM --rank 2000 --emb_reg 0.0075 --time_reg 0.01 

python learner.py --dataset ICEWS05-15 --model TeLM --rank 2000 --emb_reg 0.0025 --time_reg 0.1

python learner.py --dataset yago11k --model TeLM --rank 2000 --emb_reg 0.025 --time_reg 0.001

python learner.py --dataset wikidata12k --model TeLM --rank 2000 --emb_reg 0.025 --time_reg 0.0025

```


## License
TeLM is CC-BY-NC licensed, as found in the LICENSE file.

## Acknowledgement
We refer to the code of RREA. Thanks for their great contributions!

## Citation
If you use the codes, please cite the following paper:

        @inproceedings{TEAGNN,
            title = "Time-aware Graph Neural Network for Entity Alignment between Temporal Knowledge Graphs",
            author = "Xu, Chengjin  and
              Su, Fenglong  and
              Lehmann, Jens",
            booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
            month = nov,
            year = "2021",
            address = "Online and Punta Cana, Dominican Republic",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.emnlp-main.709",
            doi = "10.18653/v1/2021.emnlp-main.709",
            pages = "8999--9010",
        }
