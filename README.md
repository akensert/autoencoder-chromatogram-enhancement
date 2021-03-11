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
python train.py --GPU=True && \
python evaluate.py --N=10000 --T=white && \
python evaluate.py --N=10000 --T=pink
```
or `python train.py -GPU=False` if GPU isn't available. Notice that the training will take very long on a CPU (possibly several days).

**(1) generate_dataset.py.** Generates three datasets: train, test\_white and test\_pink to `input/`. To reduce the number of training examples, add the flag `--N=1000` to generate 1000 training examples instead of the default 190000. And to generate second derivatives instead of the default third derivatives add the flag `--der_order=2`; this will result in the autoencoder optimizing itself towards approximating second derivatives instead of third derivatives. Based on the number of CPU threads, the generation of datasets may take anything from 1 and 30 minutes.

**(2) train.py.** Trains autoencoder on train data (chromatograms in `input/training/*`). Takes about 4 hours to run with an RTX 2070 GPU; and depending on the performance of the CPU, training the autoencoder on a CPU may take anything from <1 day to a couple of days.

**(3) evaluate.py.** Evaluates autoencoder + the other denoising methods on test\_white and test\_pink chromatograms (chromatograms in `input/test_white/` and `input/test_pink/` respectively). This step may take up to a few hours to run. Reduce time by lowering the number of test examples; i.e. set flag `--N=100` to evaluate on 100 test examples instead of the default/maximum of 10000. Notice, the hyperparameter values of the denoising methods were obtained via the hyperparameter grid-search (see `hyperparameter_search.py`). The evaluation runs on a single CPU thread, thus taking approximately an hour to run.
