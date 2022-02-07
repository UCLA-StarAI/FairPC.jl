# Learn Fair PSDD

## Installation

1. Julia version 1.7

2. Run commands with flag `--project` will automatically use the packages specified in `Project.toml`. See belows scripts for examples.

### Files

```
  baselines/    Python scripts to reproduce `Reweight`, `Reduction` and `FairLR`.
  bin/          Runnable julia srcipts (see below).
  circuits/     Learned circuits in experiments.
  data/         Datasets used in the experiments.
  scripts/      Helper files to generate experiments scripts.
  src/          The source code for the algorithm.
  Project.toml  This file specifies required julia environment.
  README.md     This is this file.
```

## Experiments

### Usage

- Run `bin/learn.jl` with `--help` argument to see the usage message. 
Most of the options have default values. The following are some arguments need to be manully set:

```
positional arguments:
  dataset               dataset name, in {compas, adult, german, synthetic}
optional arguments:
  --sensitive_variable  sensitive variable of current data set, e.g.,{Ethnic_Code_Text_, sex, S}
  --fold                fold id for k-fold cross validation, in [1:10]
  --struct_type         indicate structure constrains of probability distributions, in {FairPC, TwoNB, NlatPC, LatNB}
  --num_X               number of non sensitive features in synthetic data set setting, in [10:30]
```

- Some sample scripts

```
$  julia --project bin/learn.jl compas --exp-id 1  --dir "exp/compas/1" --struct_type "FairPC"  --sensitive_variable "Ethnic_Code_Text_"  --fold 1 
$  julia --prroject bin/learn.jl synthetic --exp-id 2  --dir "exp/synthetic/2" --struct_type "FairPC"  --num_X 10  --sensitive_variable "S"  --fold 1 
```

- To generate multiple scripts and run batches of experiments in parallel, run the following for real-world dataset and synthetic dataset respectively:

``` 
$ julia --project bin/gen_exp.jl scripts/json/realworld-fair.json 
$ julia --project bin/gen_exp.jl scripts/json/synthetic-fair.json
```
you can also change `dir` in file `*.json` to the output directory you want.

### Baselines
- For `TowNB`, `LatNB`, and `NlatPC`, see above.
- For `Reduction`, `Reweight`, and `FairLR` methods, run `fair_reduction.py`, `reweight.py` or `fair_lr.py` respectively(the first two in `python3` and the last in `python2`) in directory `.\baselines` and following arguments:
```
# usage is the same as above
positional arguments:
  dataset
optional arguments:
  --fold
  --num_X
```
- Some sample scripts
```
# Reweight
$ python3 reweight.py compas --fold 1
$ python2 fair_lr.py synthetic --fold 1 --num_X 30
$ python3 fair_reduction.py german --fold 2
```
- To generate batches scripts, run:
```
$ julia bin/gen_exp.jl scripts/json/baselines.json --set_id 0 --cmd python3 -b fair_reduction.py
$ julia bin/gen_exp.jl scripts/json/baselines.json --set_id 0 --cmd python3 -b reweight.py
$ julia bin/gen_exp.jl scripts/json/fair-baselines.json --set_id 0 --cmd python -b fair_lr.py
```