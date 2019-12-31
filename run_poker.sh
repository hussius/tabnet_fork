# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

#virtualenv -p python3 .
#source ./bin/activate

#pip install tensorflow
#pip install -r requirements.txt

python train_tabnet.py --csv-path data/poker_train.csv --target-name CLASS --task classification --categorical-features S1,S2,S3,S4,S5 --feature-dim 24 --output-dim 8 --lambda-sparsity 0.001 --virtual-batch-size 256 --batch-momentum 0.8 --n_steps 4 --gamma 1.5 --lr 0.02 --decay-every 10000 --max-steps 71000
