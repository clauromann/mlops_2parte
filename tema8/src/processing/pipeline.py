import pandas as pd

from processing.cleaning import (
    drop_cabin,
    drop_redundant_columns,
    impute_age_by_sex_and_class,
    impute_embarked,
    impute_fare,
)
from processing.features import add_family_features, add_title_feature


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    processed = drop_cabin(df)
    processed = impute_embarked(processed)
    processed = impute_fare(processed)
    processed = impute_age_by_sex_and_class(processed)
    processed = add_title_feature(processed)
    processed = add_family_features(processed)
    processed = drop_redundant_columns(processed)
    return processed
