# Command-line interface for "TabNet: Attentive Interpretable Tabular Learning"

This is a totally unofficial fork of Google Research's TabNet repo at https://github.com/google-research/google-research/tree/master/tabnet. 

The original code accompanies a manuscript, [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442),
which was written by Sercan O. Arik and Tomas Pfister.

Please refer to the original repository for the Forest Cover experiments described in the paper.

## How do I train a TabNet model?

After having installed the dependencies, perhaps into a virtual environment, with

`pip install -r requirements.txt`

Run a command like

``` 
python train_tabnet.py --csv-path data/adult.csv \
                       --target-name <=50K \
                       --categorical-features workclass,education,marital.status,occupation,relationship,race,sex,native.country
                       --task classification 
``` 

There are also many other parameters that can be specified, e.g. 

```
python train_tabnet.py --csv-path data/adult.csv \
                       --target-name "<=50K" \
                       --categorical-features workclass,education,marital.status,occupation,relationship,race,sex,native.country \
                       --task classification
                       --val-frac 0.2 \
                       --test-frac 0.1 \
                       --model-name tabnet_adult_census \
                       --tb-log-location adult_census_logs \
                       --emb-size 1 \ 
                       --feature_dim 16 \
                       --output_dim 16 \
                       --n_steps 5 \
                       --lambda-sparsity 0.0001 \
                       --gamma 1.5 \
                       --lr 0.02 \
                       --batch-momentum 0.98 \
                       --batch-size 4096 \
                       --virtual-batch-size 128 \
                       --decay-every 500 \
                       --max-steps 3000
```

Because there are so many possible parameters, some "ready-made" configurations are available as shell scripts   
                    
## Hyperparameter optimization

Run a command like

`python opt_tabnet.py --csv-path data/poker_train.csv --target-name CLASS --categorical-features S1,S2,S3,S4,S5`

Note that you should probably change the parameters _max_steps_ and _early_stop_steps_ manually depending on the dataset.

## What's different in this repo?

I wanted to be able to run TabNet in a more flexible way, where I would not have to worry about manually changing the code as
in the original implementation. I also wanted to be able to train TabNet models directly on CSV files. This means that you 
should only run this code if your dataset fits in RAM. Otherwise, it is better to use the author's original CSV data loader
(which is still retained in this version).

The code here has functionality to train TabNet models from the command line, and includes some toy datasets for reference:
Adult Census, Poker Hands, and AutoMPG (the latter one is for regression rather than classification.)

Caveats:

- I am not sure I have implemented the regression loss correctly, as there was no example of that in the original code.
- The default size of the embedding layers is one (!). This was the case in the original code and one of the authors
explained to me that it is done for the sake of interpretability.