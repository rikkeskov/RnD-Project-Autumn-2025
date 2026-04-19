from tsai.all import *
from data_preprocessing import generate_X_y

DATASET = "BIDMC32"
DIMENSION = "AVR"

WINDOW_LENGTH = 100
HORIZON = 1
STRIDE = None

test_file = "/" + DATASET + "_TEST_dim" + DIMENSION + ".csv"

def infer():
    PATH = Path("./models/regression_BIDMC32_dimAVR_rmse0.04831_InceptionTime.pkl")
    learn = load_learner(PATH, cpu=False)

    X, y = generate_X_y(test_file, WINDOW_LENGTH, HORIZON, STRIDE)
    print("X shape before DataLoader:", X.shape)
    print("y shape before DataLoader:", y.shape)

    probas, _, preds = learn.get_X_preds(X, bs=16)
    skm.root_mean_squared_error(y, preds)

if __name__ == "__main__":
    infer()