# Command-line interface for "TabNet: Attentive Interpretable Tabular Learning"

This is a totally unofficial fork of Google Research's TabNet repo at https://github.com/google-research/google-research/tree/master/tabnet. 

The original code accompanies a manuscript, [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442),
which was written by Sercan O. Arik and Tomas Pfister.

Please refer to the original repository for the Forest Cover experiments described in the paper.

## What's different in this repo?

I wanted to be able to run TabNet in a more flexible way, where I would not have to worry about manually changing the code as
in the original implementation. I also wanted to be able to train TabNet models directly on CSV files. This means that you 
should only run this code if your dataset fits in RAM. Otherwise, it is better to use the author's original CSV data loader
(which is still retained in this version).

The code here has functionality to train TabNet models from the command line, and includes some toy datasets for reference:
Adult Census, Poker Hands, and AutoMPG (the latter one is for regression rather than classification.)

Caveats:

- I am not sure I have implemented the regression loss correctly, as there was no example of that in the original code.
- I have not implemented early stopping or similar mechanisms, but you can watch the training metrics in TensorBoard and it
should be quite straightforward to monitor the validation loss and stop the training. 
- The default size of the embedding layers is one (!). This was the case in the original code and one of the authors
explained to me that it is done for the sake of interpretability.