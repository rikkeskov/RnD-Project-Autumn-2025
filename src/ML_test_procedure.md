# TEST PROCEDURE
1. split in test and train (and validation?)
- This split must be by choosing a before/after timestamp i.e. no random splitting

(Iteratively)
2. choose window size (key hyperparameter)
- choose by doing:
- - Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots
- - Domain knowledge (e.g., seasonality or periodicity)
- - Cross-validation over different lag values
3. choose overlap size
4. choose forecast horizon i.e. number of target values = y

5. Train on training data
6. test on uncompresssed test data
7. test on compressed test data