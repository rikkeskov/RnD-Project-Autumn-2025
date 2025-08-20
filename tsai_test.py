"""Testing of TSAI"""
import numpy
from tsai.all import *
import sklearn.metrics as skm

print(get_UCR_univariate_list())

dsid = 'AppliancesEnergy' 
PATH = Path('./models/Regression.pkl')

X, y, splits = get_regression_data(dsid, split_data=False)
tfms  = [None, [TSRegression()]]
batch_tfms = TSStandardize(by_sample=True, by_var=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse])
learn.fit_one_cycle(50, 1e-2)

PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)
learn = load_learner(PATH, cpu=False)
raw_preds, target, preds = learn.get_X_preds(X[splits[1]])
np.save('test.npy', preds)
print('predictions saved, preds.shape')

print(raw_preds)
print(target)
print(preds)