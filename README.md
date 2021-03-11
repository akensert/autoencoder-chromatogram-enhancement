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
or `python train.py --GPU=False` if GPU isn't available. Notice that the training will take very long on a CPU (possibly several days).

**(1) generate_dataset.py.** Generates three datasets: train, test\_white and test\_pink to `input/`. To reduce the number of training examples, add the flag `--N=1000` to generate 1000 training examples instead of the default 190000. And to generate second derivatives instead of the default third derivatives add the flag `--der_order=2`; this will result in the autoencoder optimizing itself towards approximating second derivatives instead of third derivatives. Based on the number of CPU threads, the generation of datasets may take anything from 1 and 30 minutes.

**(2) train.py.** Trains autoencoder on train data (chromatograms in `input/training/*`). Takes about 4 hours to run with an RTX 2070 GPU; and depending on the performance of the CPU, training the autoencoder on a CPU may take anything from <1 day to a couple of days.

**(3) evaluate.py.** Evaluates autoencoder + the other denoising methods on test\_white and test\_pink chromatograms (chromatograms in `input/test_white/` and `input/test_pink/` respectively). Notice, the hyperparameter values of the denoising methods were obtained via the hyperparameter grid-search (see `hyperparameter_search.py`). The evaluation runs on a single CPU thread, thus taking approximately an hour to run. Reduce run-time by lowering the number of test examples; i.e. set flag `--N=100` to evaluate on 100 test examples instead of the default/maximum of 10000.

### Shortcut: Try denoising a chromatogram yourself right away with a pretrained autoencoder:
Run from terminal:
```
python denoise.py --path='../data/chromatogram_ISO.csv'
```
Feel free to include your own csv file, though a few chromatograms are supplied in `data/` (e.g. `chromatogram_ISO.csv`).<br><br>
*Procedure (script):*<br>
1. Reads the chromatogram from a csv file given the path (notice, the csv should contain two columns (comma separated, no header): column 1 being be the time and column 2 being the strength of the signal (e.g. mAU))
2. Lets the trained autoencoder denoise it
3. Writes the denoised chromatogram to a new csv file (same path as the input but with the extension `.smooth.csv`)
<br><br>
*Limitations:*<br>

1. The autoencoder was trained on peak heights of 25-250, thus if the average peak height of your chromatogram is <50 or >500, the results of the denoising might be bad. To resolve this, multiply each element of the signal column by some constant (e.g. if the average peak height is ~10, multiply the entire signal by ~30).
2. Similarily, as the autoencoder was trained on chromatograms with certain ranges of values of SNR, peak width, number of peaks, etc., the results of the denoising might be bad if your chromatogram is outside those ranges (see `generate_dataset.py` for more information on these ranges of values)
