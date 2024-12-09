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


def get_speech_duration(time_series: np.array, sr: int) -> np.float64:
    return np.float64(len(time_series)) / sr


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
        dialect_region in the form (DR<IDX>, DR Name)
        speaker_info (speaker_id , sex denoted by `m` or `f`)
    """

    # Get the prefixes for each part of the path, ignoring `./` at the beginning
    timit_rel_path = timit_rel_path.lstrip("./")
    splits = timit_rel_path.split("/")

    dialect_region = splits[1]
    dialect_region_name = DIALECT_REGION_LOOKUP[dialect_region]

    # Speaker ID in 3rd position
    speaker_id = splits[2]

    # sex is denoted by the first letter of splits[2]
    sex = splits[2][0]
    return (dialect_region, dialect_region_name), (speaker_id, sex)


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


def get_replace_ending(path: str, new_extension: str) -> str:
    # Get the base name of the path
    base_name, _current_extension = os.path.splitext(path)

    # Remove the . if it exists
    new_extension = new_extension.lstrip(".")

    # concatenate the `new_extension`
    return f"{base_name}.{new_extension}"


def get_sentence_info(path: str) -> tuple[str, str]:
    """
    Takes in an absolute path to a TIMIT file and returns
    the sentence ID (S<SENTENCE_TYPE><ID#>) and 
    sentence_type (2 letter code either `SA`, `SX`, `SI`)
    """
    file_name = os.path.basename(path)
    sentence_id, _file_extension = os.path.splitext(file_name)
    sentence_type = sentence_id[:2]

    return sentence_id, sentence_type
