# Helper functions to extract metadata from the timit database
import numpy as np
import pandas as pd
import os

DIALECT_REGION_LOOKUP = {
    "DR1":  "New England",
    "DR2":  "Northern",
    "DR3":  "North Midland",
    "DR4":  "South Midland",
    "DR5":  "Southern",
    "DR6":  "New York City",
    "DR7":  "Western",
    "DR8":  "Army Brat (moved around)"
}


def get_speech_duration(time_series: np.array, sr: int) -> np.float16:
    return np.float16(len(time_series))/sr


def get_timit_path(abs_path: str, base_path: str) -> str:
    """
    Remove the starting overlapping portion of the base path 
    so that only the relative path is returned.

    ex: 
    base_path = "../data/input_data/TIMIT-Database/TIMIT"
    abs_path = "../data/input_data/TIMIT-Database/TIMIT/TRAIN/DR4/MMDM0/SI681.wav"
    get_timit_path(abs_path, base_path) = TRAIN/DR4/MMDM0/SI681.wav
    """
    return os.path.relpath(abs_path, base_path)


def get_speaker_info(timit_rel_path: str) -> tuple[tuple[str, str], str]:
    """
    Takes in a relative timit path in the form
        `USAGE/DIALECT/SEX+SPEAKER_ID/SENTENCE_ID.FILE_TYPE`
        taken from: https://catalog.ldc.upenn.edu/docs/LDC93S1/timit.readme.html

        i.e. TRAIN/DR4/MMDM0/SI681.wav

    returns:
        dialect_region in the form [DR<IDX>, DR Name]
        sex denoted by `m` or `f` 
    """

    # Get the prefixes for each part of the path, ignoring `./` at the beginning
    timit_rel_path = timit_rel_path.lstrip("./")
    splits = timit_rel_path.split("/")

    dialect_region = splits[1]
    dialect_region_name = DIALECT_REGION_LOOKUP[dialect_region]

    # sex is denoted by the first letter of splits[2]
    sex = splits[2][0]
    return (dialect_region, dialect_region_name), sex


def get_text(abs_timit_txt_path: str) -> str:
    """
    Returns the transcription in the timit_txt_path
    """
    with open(abs_timit_txt_path, mode='r') as file:
        # Remove the first 2 strings which are the start and end frames
        line = file.readline().lstrip().rstrip()
        words = line.split(" ")[2:]
        sentence = " ".join(words)
    return sentence


def get_transcription_detail(abs_timit_transcript_path: str) -> list[dict]:
    """
    Returns the word level time-sliced transcription of the utterance.

    Note: this works for both `PHN` and `WRD` files

    ex for "her wardrobe consists only of skirts and blouses"
    [
        {'start': 2840, 'stop': 4887, 'utterance': 'her'}.
        {'start': 4887, 'stop': 14710, 'utterance': 'wardrobe'},
        ...
        {'start': 40426, 'stop': 51470, 'utterance': 'blouses'}
    """
    df = pd.read_csv(abs_timit_transcript_path, sep=r"\s+", names=[
                     "start", "stop", "utterance"])
    return df.to_dict(orient="records")
