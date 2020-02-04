# Assignment 2

- Name: Shreyas Shankar
- Student ID: 20602181

## Dependencies

- python3
- torch (PyTorch v1.4.0)
- scikit-learn
- numpy
- pandas
- json
- argparse
- os
- tqdm
- matplotlib.pyplot
- seaborn

## Running `main.py`

If all the dependencies and supporting files are cloned from this repo, `main.py` can be directly run from within this repo with the following sparse command:

```
python3 main.py
```
With command line parameters passed, the execution can also be:
```
python3 main.py -p ./files/param.json -d ../Problem/even_mnist.csv -r ./results -v True -c True
```
Flags `-p` and `-d` are necessary to provide only if the program is run with files different from those provided within this repo. Flag `-r` is necessary if you wish to save the resultant loss plot. Otherwise it will only display without saving.
