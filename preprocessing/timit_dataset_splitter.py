import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, Audio


def stratify_timt_dataset(
        dataset: Dataset,
        validation_size: np.float64 = 0.5,
        test_size: np.float64 = 0.5,
        seed: int = 42
) -> tuple[Dataset, Dataset]:
    """
    Splits the timit_dataset stratified on the `sex` and `dialect region`

    returns:
        - validation_dataset
        - test_dataset
    """
    # Step 1) Get the HF dataset as Dataframe
    orig_df: pd.DataFrame = dataset.to_pandas()

    # Step 2) Create strata criteria
    speaker_metadata = orig_df.groupby("speaker_id").agg({
        "speaker_sex": "first",
        "dialect_region": "first",
    }).reset_index()

    speaker_metadata["strata"] = \
        speaker_metadata["speaker_sex"] \
        + "_" \
        + speaker_metadata["dialect_region"]

    # Step 3) Create split labels
    validation_split, test_split = train_test_split(
        speaker_metadata,
        train_size=validation_size,
        test_size=test_size,
        stratify=speaker_metadata["strata"],
        random_state=seed
    )

    # Step 4) Apply split to dataset
    validation_df = orig_df[
        orig_df["speaker_id"].isin(validation_split["speaker_id"])
    ]
    test_df = orig_df[
        orig_df["speaker_id"].isin(test_split["speaker_id"])
    ]

    # Step 5) Cleanup Pandas Conversions
    
    # Remove the extra column index used for grouping
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Convert from pandas back to HF
    validation_dataset = Dataset.from_pandas(validation_df).cast_column("audio", Audio())
    test_dataset = Dataset.from_pandas(test_df).cast_column("audio", Audio())

    return validation_dataset, test_dataset
