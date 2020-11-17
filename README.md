## Deep denoising autoencoder for chromatogram enhancement


### How To

To run the full pipeline (including dataset generation, autoencoder training and evaluation), run the following from terminal:
```
python generate_dataset.py && \
python train.py --GPU=False && \
python evaluate.py --N=10000 --T=white && \
python evaluate.py --N=10000 --T=pink
```
or `python train.py -GPU=True` if GPU is available.

**(1) generate_dataset.py.** Generates three datasets: train, test\_white and test\_pink to `input/`

**(2) train.py.** Trains autoencoder on train data (chromatograms in `input/training/*`). Takes about 30 minutes with an RTX 2070 GPU.

**(3) evaluate.py.** Evaluates autoencoder + the other algorithms on test\_white and test\_pink chromatograms (chromatograms in `input/test_white/*` and `input/test_pink/*` respectively). Takes about 10 hours to run. Reduce time by lowering the number of test examples; i.e. set --N=100 to evaluate on 100 test examples instead of the maximum 10000.


### Notebooks
Try out the jupyter notebooks to better understand (e.g. by visualizing) the different steps from above.
