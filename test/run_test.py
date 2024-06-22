import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import random

from sklearn.metrics import mean_squared_error, r2_score
import test_data 
from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    X_test, y_test = sample_input_data
    #print(y_test.shape)
    y_pred = make_prediction(input_data=X_test)["predictions"]
    #print(y_pred.shape)

    print("R2 score:", r2_score(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))

test_make_prediction(test_data.test_data())