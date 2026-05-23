import pandas as pd

from processing.features import (
    add_family_features,
    add_title_feature,
    categorize_family_size,
    extract_title,
    normalize_title,
)


def test_extract_and_normalize_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert normalize_title("Mlle") == "Miss"
    assert normalize_title("Mme") == "Mrs"
    assert normalize_title("Dr") == "Rare"


def test_add_title_feature_creates_expected_values():
    df = pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Heikkinen, Miss. Laina",
                "Minahan, Dr. William Edward",
            ]
        }
    )

    result = add_title_feature(df)

    assert result["Title"].tolist() == ["Mr", "Miss", "Rare"]


def test_categorize_family_size():
    assert categorize_family_size(1) == "Alone"
    assert categorize_family_size(2) == "Small"
    assert categorize_family_size(4) == "Small"
    assert categorize_family_size(5) == "Large"


def test_add_family_features_creates_size_and_category():
    df = pd.DataFrame(
        {
            "SibSp": [0, 1, 3],
            "Parch": [0, 1, 2],
        }
    )

    result = add_family_features(df)

    assert result["Family_size"].tolist() == [1, 3, 6]
    assert result["Family_size_cat"].tolist() == ["Alone", "Small", "Large"]
