# Leave-One-Out Stable Conformal Prediction

This contains the implementation and experiments for the paper titled **"Leave-One-Out Stable Conformal Prediction"**, submitted to ICLR 2025.

## Requirements
- Python 3.x
- Libraries: numpy, pandas, cvxpy, scikit-learn, matplotlib, seaborn

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Running the Experiments

To reproduce the results in Section 4, run:

```bash
python simul.py
```

To reproduce the results in Section 5, run:

```bash
python real.py
```

To reproduce the results in Section 6, please download the dataset from Kaggle:

[Factors Affecting Campus Placement](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)

Once downloaded, rename the file to `recruit.csv` and place it in the folder named 'data' of this repository.

After that, run:

```bash
python recruit.py
```
