# Leave-One-Out Stable Conformal Prediction

This repository contains the implementation and experiments for the paper titled **"Leave-One-Out Stable Conformal Prediction"**, submitted to ICLR 2025.

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

To reproduce the results in Section 6, please download the dataset from Kaggle:

[Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)

Once downloaded, rename the file to `recruit.csv` and place it in the root directory of this repository (the same level as this `README.md` file).

After that, run the following command to execute the experiments:

```bash
python Recruit.py
```
