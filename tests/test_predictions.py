import sys
import warnings
import numpy as np
from sklearn.metrics import accuracy_score

# Import the make_prediction function from your module
from bikeshare_model.predict import make_prediction

def test_make_prediction(sample_input_data):

    # Given
    expected_no_predictions = len(sample_input_data[1]) 

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data[1]
    accuracy = accuracy_score(_predictions, y_true)