# Ridge + LASSO on SECOM
Visualizing how L1/L2 penalties changes the least squares fit on semiconductor manufacturing data, using OLS as the baseline and then building Ridge/LASSO equations from scratch.

# Data/Preprocessing
SECOM has 1567 samples with a ~6.6% (104) failure rate. The 'secom_labels.csv' uses '-1' for pass and '+1' for fail and we exclude the timestamps. The 'secom.csv' and 'secom_labels.csv' are joined together.

We use an 80/20 training-testing split with seed 42, assume a >50% missing column rate must mean its excluded and missing data imputed using medians. We also add a column of 'ones' considering the intercept.

# Methods
OLS contains an objective function that minimizes the squared error.
The 'np.linalg.lstsq' method accounts for the rank deficiency (only 446 out of the 451 cleaned columns were independent).

Ridge solves '(X.T @ X + λD) β = X.T @ y' with 'np.linalg.solve'
Our λ sweeps from 10E-3 to 10E12.

Our objective for Ridge targets the minimization of the sum of mean squared errors + the L2 norm of λ

We plot the mean square errors against λ.

# Run

Install the following dependencies
'python -m pip install numpy pandas scikit-learn matplotlib'
'python main.py'

# Citation

McCann, M., & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. [Dataset and license](https://archive.ics.uci.edu/dataset/179/secom) · [DOI](https://doi.org/10.24432/C54305)
