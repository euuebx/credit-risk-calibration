# Credit Default Prediction and Probability Calibration

A practical, reproducible project using real historical records from the UCI Machine Learning Repository. It compares logistic regression and random forest, then evaluates whether calibration improves predicted default probabilities.

**No synthetic data is used to train or evaluate the models.** The project downloads the authentic UCI dataset. Unit tests use tiny hand-constructed examples only to test mathematical behaviour; they are not used as project results or model training data.

## Research question

Can a model rank borrowers by default risk, and are its predicted probabilities close to observed default frequencies? Ranking and calibration are different properties: a model can rank borrowers well while systematically overstating or understating probabilities.

## Real data source

- Dataset: **Default of Credit Card Clients**
- Provider: UCI Machine Learning Repository
- Official page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Raw file is downloaded from UCI's public dataset archive at runtime.
- Historical Taiwan credit-card client data, with payment status, bill amounts, payment amounts and demographic/credit-limit fields. The target is default payment next month.
- The downloaded raw file's SHA-256 checksum, sample sizes and default rate are recorded in `outputs/run_metadata.json`.

This is an older, location-specific dataset. It is useful for a student modelling project but should not be presented as representative of current Irish or European credit portfolios. Check UCI's data licence and documentation before redistributing the raw file.

## Modelling design

1. **Split data** into stratified train (60%), calibration (20%) and test (20%) sets.
2. **Preprocess inside a pipeline** so imputers, scaling and one-hot encoding are fitted only on training rows.
3. **Train two models:** logistic regression as an interpretable baseline; random forest as a nonlinear comparator.
4. **Calibrate on a separate held-out set** using sigmoid (Platt-style) calibration by default. The base estimator is frozen while calibration is fitted.
5. **Evaluate once on the held-out test set**, both before and after calibration.
6. **Report ranking, probability quality, and threshold-dependent metrics** rather than accuracy alone.

The default split is random and stratified because the dataset is not a clean modern time-series loan origination panel. It is a baseline evaluation, not a simulation of a future deployment. Avoid claiming temporal generalisation.

## Mathematical framework

Let \(Y_i\in\{0,1\}\) indicate default and \(X_i\) be borrower attributes. The model estimates

\[
\hat p_i = \widehat{\Pr}(Y_i=1\mid X_i).
\]

Logistic regression uses

\[
\hat p_i=\frac{1}{1+\exp(-(\beta_0+X_i^\top\beta))}.
\]

The coefficients are estimated by minimising regularised log loss. A random forest provides a nonlinear comparison by averaging class probabilities over trees.

### Calibration

A calibrated model should satisfy approximately

\[
\Pr(Y=1\mid \hat p\approx q)\approx q.
\]

For example, among cases assigned about 20% default probability, the observed default rate should be close to 20% over a sufficiently large group. Sigmoid calibration learns a logistic mapping from raw model scores to outcomes using the separate calibration set.

### Evaluation

- **ROC-AUC:** ability to rank defaults above non-defaults.
- **Average precision / PR-AUC:** precision-recall summary, informative with imbalanced targets.
- **Brier score:** mean squared probability error,

\[
\mathrm{Brier}=\frac{1}{n}\sum_{i=1}^n(\hat p_i-y_i)^2.
\]

- **Log loss:** penalises confident incorrect probabilities.
- **Expected calibration error (ECE):** weighted average difference between predicted probability and observed event rate across bins.
- **Precision, recall and F1:** reported at the stated threshold only. The default 0.5 threshold is not automatically a sensible credit decision threshold.

Lower Brier, log loss and ECE are better; higher ROC-AUC and average precision are generally better. Calibration metrics depend on binning and sample size, so inspect the reliability diagram too.

## Install and run

Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
credit-risk
```

First run needs internet access to download the UCI archive. The raw dataset is cached locally; use `credit-risk --refresh` to download it again. If Excel reading fails, install a compatible Excel engine (`xlrd` is typically required for legacy `.xls` files):

```bash
pip install xlrd
```

## Output files

- `outputs/metrics.csv`: metrics for each model before and after calibration.
- `outputs/evaluation.png`: reliability curves and ranking metrics.
- `outputs/*_test_probabilities.csv`: true labels and out-of-sample probabilities.
- `outputs/*_confusion_matrix.csv`: threshold-specific confusion matrices.
- `outputs/run_metadata.json`: data checksum, sample sizes, seed and caveats.
- `models/`: serialized base and calibrated estimators.

## Caveats and next improvements

- The dataset is historical and geographically specific.
- The random split is not a temporal validation design.
- This is not a fairness audit or lending decision system.
- The 0.5 threshold is illustrative; real decisions require explicit costs and policy constraints.
- ECE varies with binning; report reliability plots and proper scoring rules alongside it.
- For a stronger second version: use nested cross-validation, bootstrap confidence intervals, subgroup performance and calibration checks, and an explicit cost-based threshold analysis. Avoid selecting the best method using the test set.

## Interview explanation

> I used real historical credit-default records from UCI. I built logistic regression as an interpretable baseline and compared it with a random forest. I kept training, calibration and test sets separate to reduce leakage, then compared ranking metrics with probability-quality metrics. The key finding is whether calibration improved Brier score, log loss and reliability without assuming the more complex model was automatically better.

## CV bullet templates

Fill these with values from your actual `outputs/metrics.csv` after a successful run:

- Built a Python credit-default modelling pipeline on **[N]** real UCI records, comparing logistic regression and random forest with leakage-aware preprocessing.
- Evaluated default-risk ranking using **[ROC-AUC] ROC-AUC** and **[average precision] average precision**, reporting performance on a held-out test set.
- Tested sigmoid probability calibration, changing Brier score from **[before]** to **[after]** and ECE from **[before]** to **[after]**; documented dataset limitations and evaluation caveats.

Never fill in bracketed values with estimates or invented results.
