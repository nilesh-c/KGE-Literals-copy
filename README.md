# KGE-Literals

Link prediction with literals in PyTorch.

## Getting Started

1. Install miniconda <http://conda.pydata.org/miniconda.html>
2. Do `conda env create`
3. Enter the env `source activate kga`
4. Install [Pytorch v0.3+](https://github.com/pytorch/pytorch#installation)

## Preparing new experiments

1. Create your model in `kga/models/literals.py`.
2. Create experiment script at `experiments/{fb15k,yago,ml}` dir. See baseline files `run_multitask_{fb15k,yago,ml}` for reference and use it as template.
    * The script has to implements all of the variables as in the `run_multitask_{fb15k,yago,ml}`'s argument parsers.
    * It is very important to use proper model name when saving the model file. Consult the reference above.
3. Run the experiment, e.g. `nohup python -u experiments/yago3-10/run_multitask_yago.py --use_gpu --log_interval -1 --nepoch 300 --lr_decay_every 100 --k 100 --mbsize 200 --lr 1e-3 --weight_decay 5e-4 &> mtkgnn_yago_lr1e-3_wd5e-4 &`
    * In the example above, tt will save a model in `models/yago/mtkgnn_yago_lr0.001_wd0.0001`
4. After finished training, test the resulting model, e.g. `python experiments/yago3-10/run_multitask_yago.py --use_gpu --k 100 --test --test_model mtkgnn_yago_lr0.001_wd0.0001`
    * This will load the models from the previous training.
    * It will print all of the evaluation metrics over the test set.

## Dependencies

1. Python 3.5+
2. PyTorch 0.2+
3. Numpy
4. Scikit-Learn
5. Pandas
