## Deep denoising autoencoder for chromatogram enhancement

### Requirements
* Python >= 3.6
* TensorFlow >= 2.0
* Pip (package manager)
* Third party Python packages found in `requirements.txt`. To install these packages (including TensorFlow), run from terinal: `pip install -r requirements.txt`

### How to run
To run the full pipeline (including dataset generation, autoencoder training and evaluation), run the following from terminal:
```
python generate_dataset.py && \
python train.py --GPU=False && \
python evaluate.py --N=10000 --T=white && \
python evaluate.py --N=10000 --T=pink
```
or `python train.py -GPU=True` if GPU is available.

**(1) generate_dataset.py.** Generates three datasets: train, test\_white and test\_pink to `input/`. To reduce the number of training examples, add the flag `--N=1000` to generate 1000 training examples instead of the default 190000. And to generate second derivatives instead of the default third derivatives add the flag `--der_order=2`; this will result in the autoencoder optimizing itself towards approximating second derivatives instead of third derivatives.

**(2) train.py.** Trains autoencoder on train data (chromatograms in `input/training/*`). Takes about 30 minutes with an RTX 2070 GPU.

**(3) evaluate.py.** Evaluates autoencoder + the other algorithms on test\_white and test\_pink chromatograms (chromatograms in `input/test_white/*` and `input/test_pink/*` respectively). Takes about 10 hours to run. Reduce time by lowering the number of test examples; i.e. set --N=100 to evaluate on 100 test examples instead of the maximum 10000. Notice, the hyperparameter values of the denoising methods were obtained via the hyperparameter grid-search (see `hyperparameter_search.py`)

### Notebooks
Try out the jupyter notebooks to better understand (e.g. by visualizing) the different steps from above.
