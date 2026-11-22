import pandas as pd
from sklearn.preprocessing import LabelEncoder


REQUIRED_COLUMNS = {"Age", "Fare", "Sex", "SibSp", "Parch", "Embarked"}


def _validate_columns(dataframe, name):
    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"{name} не содержит обязательные колонки Titanic: {missing}. "
            f"Получены: {list(dataframe.columns)}"
        )


def preprocess_titanic_data(train_df, test_df):
    _validate_columns(train_df, "train_df")
    _validate_columns(test_df, "test_df")

    combined = pd.concat([train_df, test_df], ignore_index=True)

    for column in ["Age", "Fare", "SibSp", "Parch"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
        median = combined[column].median()
        if pd.isna(median):
            raise ValueError(
                f"Невозможно заполнить колонку {column}: все значения отсутствуют."
            )
        combined[column] = combined[column].fillna(median)

    embarked_mode = combined["Embarked"].mode(dropna=True)
    if embarked_mode.empty:
        raise ValueError("Невозможно заполнить Embarked: все значения отсутствуют.")
    combined["Embarked"] = combined["Embarked"].fillna(embarked_mode.iloc[0])

    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

    encoders = {}
    for column in ["Sex", "Embarked"]:
        encoder = LabelEncoder()
        combined[column] = encoder.fit_transform(combined[column].astype(str))
        encoders[column] = encoder

    split_index = len(train_df)
    processed_train = combined.iloc[:split_index].copy()
    processed_test = combined.iloc[split_index:].copy()

    if "Survived" in processed_test.columns:
        processed_test = processed_test.drop(columns=["Survived"])

    return processed_train, processed_test, encoders

