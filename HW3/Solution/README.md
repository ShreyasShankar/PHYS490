# Assignment 3

- Name: Shreyas Shankar
- Student ID: 20602181

## Dependencies

- python3
- torch (PyTorch v1.4.0)
- numpy
- json
- argparse
- os
- tqdm
- matplotlib.pyplot

## Running `main.py`

If all the dependencies and supporting files are cloned from this repo, `main.py` can be directly run from within this repo with the following sparse command:

```
python3 main.py
```
With command line parameters passed, the execution can also be:
```
python3 main.py -p ./files/param.json -d ../Problem/in.txt -r ./results -v True
```
For more information on the optional parameters, enter the command:
```
python3 main.py -h
```

## Hyperparameter JSON FILE

The hyperparameter file provided, `params.json`, is located in [files](./files). It is already tuned for performance with the model. Any user-provided hyperparameter file must contain the following parameters:
* learning_rate : the learning rate of the Boltzmann machine
* num_epochs : the number of epochs for which the model will train
* display_epochs : the epoch interval for which the program will produce regular progress updates during training (only needed if verbose flag -v is not False)
