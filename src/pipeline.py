import joblib
import pandas as pd

def pipeline(df: pd.DataFrame) -> int:

    selected_col = ['city', 'homeType', 'latitude', 'longitude',
       'garageSpaces', 'hasSpa', 'yearBuilt', 'numOfPatioAndPorchFeatures',
       'lotSizeSqFt', 'avgSchoolRating', 'MedianStudentsPerTeacher',
       'numOfBathrooms', 'numOfBedrooms']
    
    df = df[selected_col].copy()

    #df.drop(columns=["uid", "description", "priceRange"], axis=1, inplace=True)

    df['hasSpa'] = df['hasSpa'].replace({True: 1, False: 0})

    path_ordinal_encoder_model = '../models/ordinal_encoder_model.joblib'
    path_clf_model = '../models/clf_model.joblib'

    ordinal_encoder_model = joblib.load(path_ordinal_encoder_model)
    clf_model = joblib.load(path_clf_model)

    df = ordinal_encoder_model.transform(df)

    return clf_model.predict(df)[0]