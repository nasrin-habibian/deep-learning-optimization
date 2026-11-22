# Optimization Dynamics and Convergence Analysis in a Small Deep Neural Network

A reproducible computational study comparing **SGD, SGD with momentum, Adam, and L-BFGS** when training the same multilayer perceptron on a controlled nonlinear classification problem.

This repository is designed as a learning and research-practice project for scientific computing. It provides a transparent experimental pipeline, records training diagnostics, and creates plots and tables from actual runs. It does **not** claim a novel optimizer or a new theoretical convergence result.

## Research questions

1. How do the four optimizers differ in training-loss reduction on the same model and dataset?
2. How do their gradient-norm trajectories compare?
3. How do the resulting models compare on a held-out test set?
4. How sensitive are the results to the optimizer learning-rate settings?

## Scope and limitations

- The dataset is a reproducible synthetic two-class problem generated in this repository.
- All optimizers use the same model architecture, train/test split, initialization seed, and full-batch training protocol.
- L-BFGS is a quasi-Newton method. Its `max_iter=1` setting means one outer optimizer step per recorded epoch; its internal line search may evaluate the closure multiple times.
- A single synthetic dataset cannot establish general claims about deep learning. Treat findings as observations for this controlled setting.
- Results are not included as pre-filled claims. Run the experiments to generate results on your own machine.

## Repository structure

```text
deep-learning-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── config.json
├── run_experiment.py
├── src/
│   └── dlopt/
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       ├── optimizers.py
│       ├── training.py
│       ├── experiments.py
│       └── plotting.py
└── tests/
    └── test_project.py
```

## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- Matplotlib

## Installation

Create and activate a virtual environment, then install dependencies.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the main experiment

From the repository root:

```bash
python run_experiment.py
```

The script reads `config.json`, trains each optimizer, and writes outputs to `results/`:

- `metrics.csv` — epoch-level loss, accuracy, and gradient-norm records
- `summary.csv` — final train/test metrics for each optimizer
- `loss_curves.png` — training-loss curves
- `gradient_norm_curves.png` — gradient-norm curves
- `decision_boundaries.png` — learned decision boundaries
- `experiment_metadata.json` — settings used for the run

Run with a custom configuration file:

```bash
python run_experiment.py --config path/to/your_config.json
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Experimental protocol

1. Generate a nonlinear two-class dataset using a fixed random seed.
2. Split it into training and test sets using a fixed split seed.
3. Instantiate the same MLP and load the same initial state for each optimizer.
4. Train each optimizer using the full training set at every epoch.
5. Record cross-entropy loss, classification accuracy, and the Euclidean norm of the parameter gradient.
6. Evaluate the final model on the held-out test set.
7. Save metrics, plots, and metadata.

### Important interpretation notes

- The learning-rate values are practical starting settings, not universally optimal values.
- Optimizers have different update rules and computational costs. Epoch count alone does not measure wall-clock efficiency.
- L-BFGS uses line search and may perform multiple closure evaluations per step. For rigorous compute-cost comparisons, instrument closure evaluations and wall-clock time on the target hardware.
- The gradient norm is measured before the optimizer update for SGD, momentum, and Adam. For L-BFGS, it is measured at the current parameters before the outer step.
- The experiment is deterministic where supported by the installed PyTorch build and hardware, but bitwise reproducibility is not guaranteed across platforms.

## Suggested extensions

These should be treated as future work, not as completed results:

- Repeat each experiment across several random seeds and report mean and standard deviation.
- Perform a learning-rate sensitivity study.
- Compare full-batch and mini-batch training protocols.
- Add weight decay and study its interaction with optimization.
- Test additional architectures and datasets.
- Compare wall-clock time and number of function/gradient evaluations.
- Investigate conditioning and loss-surface geometry more directly.

## How to describe this project honestly on a CV

Only after running the code and checking the outputs, a suitable description could be:

> **Optimization Dynamics in Deep Neural Networks | Python, PyTorch**  
> Implemented a reproducible comparison of SGD, momentum, Adam, and L-BFGS on a nonlinear classification task. Analysed training-loss and gradient-norm trajectories, evaluated held-out performance, and documented the experimental protocol and limitations.

Do not describe this as a novel optimizer, a theoretical convergence proof, or a publication unless you have independently completed and validated that work.

## References

- PyTorch documentation, Optimizers: https://pytorch.org/docs/stable/optim.html
- PyTorch documentation, `torch.optim.LBFGS`: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
- Kingma, D. P. and Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
- Robbins, H. and Monro, S. (1951). *A Stochastic Approximation Method*. The Annals of Mathematical Statistics.
- Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
