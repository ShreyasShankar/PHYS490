# Assignment 4

- Name: Shreyas Shankar
- Student ID: 20602181

## Dependencies

- python3
- torch (PyTorch v1.4.0)
- scikit-learn
- math
- random
- numpy
- pandas
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
python3 main.py -p ./files/param.json -d ../Problem/even_mnist.csv -r ./results -v True -c True -n 36
```
Flags `-p` and `-d` are necessary to provide only if the program is run with files different from those provided within this repo. Flag `-r` should be provided if you wish to save the generated plots in a directory of your choice. Flag `-v` toggles the verbosity of the program's output to the command line. Flag `-c` toggles usage of CUDA compute resources. Flag `-n` is used to specify the number output number images produced by the program.

For more information on the optional parameters, enter the command:
```
python3 main.py -h
```

## Hyperparameter JSON FILE

The hyperparameter file provided, `params.json`, is located in [files](./files). It is already tuned for performance with the model. Any user-provided hyperparameter file must contain the following parameters:
* test_size : the size of the dataset kept aside for testing and cross-validation
* learning_rate : the learning rate of the network's optimizer
* num_epochs : the number of epochs for which the model will train
* display_epochs : the epoch interval for which the program will produce regular progress updates during training (only needed if verbose flag -v is not False)
* batch_size : the batch size of data used for each training iteration
