import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from unittest.mock import patch, MagicMock

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import _load_raw_dataset

@pytest.fixture
def sample_data():
    data = {
        'dteday': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'weekday': ['Monday', 'Tuesday', None, None],
        'weathersit': [1, None, 2, None],
        'ordinal_category': ['low', 'high', 'medium', 'low']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_config.training_data_file)

    X = data.drop(config.model_config.target, axis=1)  # predictors
    y = data[config.model_config.target]               # target

    # Divide into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )

    return X_test, y_test