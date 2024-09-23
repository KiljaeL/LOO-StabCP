# Fast Yet Precise Conformal Prediction with Leave-One-Out Algorithmic Stability

This repository contains the implementation and experiments for the paper titled **"Fast Yet Precise Conformal Prediction with Leave-One-Out Algorithmic Stability"**, submitted to ICLR 2025.

## Contents
- `Figure/`: figures from the experiments.
- `Result/`: results of the experiments.
- `README.md`: This file.

## Requirements
- Python 3.x
- Libraries: numpy, pandas, cvxpy, scikit-learn, matplotlib, seaborn

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Experiments

To reproduce the results in Section 5.1, run:

```bash
python Synthetic.py
```

To reproduce the results in Section 5.2, run:

```bash
python Real.py
```

To reproduce the results in Section 6, download data from Kaggle:

[Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)

Please download the dataset from the link above as the name of `recruit.csv` before running the experiments.

And then, run:

```bash
python Recruit.py
```
