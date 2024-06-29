import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import calendar
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, WeekdayOneHotEncoder
from sklearn.pipeline import Pipeline

def test_weekday_imputer(sample_data):
    imputer = WeekdayImputer(variable='weekday', date_var='dteday')
    transformed_data = imputer.fit_transform(sample_data)
    assert transformed_data['weekday'].isnull().sum() == 0
    sampleDate = sample_data['dteday'][2]
    sampleDate = pd.to_datetime(sampleDate, format='%Y-%m-%d')
    expectedDay = calendar.day_name[sampleDate.weekday()][0:3]
    assert transformed_data['weekday'][2] == expectedDay



def test_weathersit_imputer(sample_input_data):
    imputer = WeathersitImputer(variable='weathersit')
    print(sample_input_data[0])
    transformed_data = imputer.fit_transform(sample_input_data[0])
    assert transformed_data['weathersit'].isnull().sum() == 0


def test_mapper(sample_data):
    mappings = {'low': 1, 'medium': 2, 'high': 3}
    mapper = Mapper(variable='ordinal_category', mappings=mappings)
    transformed_data = mapper.fit_transform(sample_data)
    assert transformed_data['ordinal_category'].dtype == int


def test_weekday_one_hot_encoder(sample_data):
    # Fill missing weekdays for the test to pass
    #sample_data['weekday'].fillna('None', inplace=True)
    sample_data.fillna({'weekday': 'None'}, inplace=True)

    encoder = WeekdayOneHotEncoder(variable='weekday')
    transformed_data = encoder.fit_transform(sample_data)
    
    print("Transformed columns:", transformed_data.columns)
    
    expected_columns = ['weekday_Monday', 'weekday_Tuesday', 'weekday_None']
    assert all(col in transformed_data.columns for col in expected_columns)
    assert 'weekday' not in transformed_data.columns


# Tests for pipeline integration and sklearn estimator compliance
def test_pipeline_integration(sample_data):
   # sample_data['weekday'].fillna('None', inplace=True)
    sample_data.fillna({'weekday': 'None'}, inplace=True)
    pipeline = Pipeline([
        ('weekday_imputer', WeekdayImputer(variable='weekday', date_var='dteday')),
        ('weathersit_imputer', WeathersitImputer(variable='weathersit')),
        ('ordinal_mapper', Mapper(variable='ordinal_category', mappings={'low': 1, 'medium': 2, 'high': 3})),
        ('weekday_encoder', WeekdayOneHotEncoder(variable='weekday'))
    ])

    pipeline.fit(sample_data)
    transformed_data = pipeline.transform(sample_data)
    
    expected_columns = ['weathersit', 'ordinal_category', 'weekday_Monday', 'weekday_Tuesday', 'weekday_None']
    assert all(col in transformed_data.columns for col in expected_columns)
    assert 'weekday' not in transformed_data.columns