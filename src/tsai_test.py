"""TSAI"""

import logging
from tsai.all import Path  # type: ignore
from .data.data_preparation import ts_to_pd_dataframe


DSID = "BIDMC32RR"
SPLIT_DATA = False
FULL_TARGET_DIR = "./data/Monash/" + DSID
PATH = "./data/Monash"

for split in ["TEST", "TRAIN"]:
    fname: Path = Path(PATH) / f"{DSID}/{DSID}_{split}.ts"
    try:
        if split == "TRAIN":
            X_train, y_train = ts_to_pd_dataframe(fname)
            # X_train = _check_X(X_train)
        else:
            X_valid, y_valid = ts_to_pd_dataframe(fname)
            # X_valid = _check_X(X_valid)
    except ValueError as inst:
        logging.error(
            "Cannot create numpy arrays for %s dataset. Error: %s", DSID, inst
        )
# np.save(f'{full_tgt_dir}/X_train.npy', X_train)
# np.save(f'{full_tgt_dir}/y_train.npy', y_train)
# np.save(f'{full_tgt_dir}/X_valid.npy', X_valid)
# np.save(f'{full_tgt_dir}/y_valid.npy', y_valid)
# np.save(f'{full_tgt_dir}/X.npy', concat(X_train, X_valid))
# np.save(f'{full_tgt_dir}/y.npy', concat(y_train, y_valid))
# PATH = Path('./models/Regression.pkl')

# X, y, splits = get_regression_data(DSID, split_data=False)
# print(X.shape, y.shape)
# check_data(X,y,splits, False)
# new_X = np.reshape(X[0][1], -1)

# with open("../data/HouseholdPowerConsumption1Row1.csv", 'w', newline='', encoding="utf-8") as csvfile:
#     writer = csv.writer(csvfile)
#     for i, value in zip(range(0, len(new_X), 1), new_X):
#         writer.writerow([i, value])
# print(y)
# numpy.savetxt("floodModeling1_test.csv", np.reshape(X[1], -1), delimiter=",")

# tfms  = [None, [TSRegression()]]
# batch_tfms = TSStandardize(by_sample=True, by_var=True)
# dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
# learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse])
# learn.fit_one_cycle(50, 1e-2)

# PATH.parent.mkdir(parents=True, exist_ok=True)
# learn.export(PATH)
# learn = load_learner(PATH, cpu=False)
# raw_preds, target, preds = learn.get_X_preds(X[splits[1]])
# np.save('test.npy', preds)
# print('predictions saved, preds.shape')

# print(raw_preds)
# print(target)
# print(preds)
