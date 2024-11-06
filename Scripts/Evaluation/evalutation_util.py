import numpy as np
from collections import Counter
from Bio import pairwise2
import matplotlib.pyplot as plt
import seaborn as sns

"""
Base Lexicon ordered in IPA Tradition of place/voice then manner
"""
BASE_LEXICON = [
    # Voicless Stops
    "p", "t", "k",
    # Voiced Stops
    "b", "d", "g",

    # Flap
    "ɾ",

    # Glottal
    "ʔ", 'h',

    # Fricatives
    'ʃ', 'ʒ', 'θ', 'ð',

    # Nasals
    'ŋ',

    # Glides
    'j', "w",

    # Vowels
    'i', 'ɪ', 'ɛ', 'æ',
    'ʌ', 'ə', 'ɚ', 'ɛ˞'
    'u', 'ʊ', 'ɔ', 'ɑ',
    'eɪ', 'aɪ', 'aʊ', 'ɔɪ', 'oʊ',
]


def remove_diacritics(transcription: list[str]) -> list[str]:
    """
    Remove vowel lengthening marker 
    TODO if there are other diacritics to be removed in the future
    """
    no_vowel_lengthening_marker = [
        token.replace("ː", "") for token in transcription]

    return no_vowel_lengthening_marker


def format_lexicon(lexicon: list[str]):
    return list(lexicon) + ["INS", "DEL"]


def create_base_confusion_matrix(lexicon: list[str] = format_lexicon(BASE_LEXICON)) -> np.ndarray:
    """
    Create the base for the confusion matrix defaulting to the size of the BASE_LEXICON
    """
    matrix = np.zeros((len(lexicon), len(lexicon)), dtype=float)
    return matrix


def build_confusion_matrix(true_pred_pairs: list[tuple[str, str]], lexicon: list[str] = format_lexicon(BASE_LEXICON)) -> np.ndarray:
    """
    Builds a confusion matrix on ((`predicted phoneme`, `true phoneme`), number of occurences)
    - Uses Biopython `pairwise2`
    - based off of https://chatgpt.com/share/672b8d54-dd9c-8010-a113-de8c683ffb0d

        We use the following formula for calculating edits (relative to the actual transcription)
            2. Deletion
                1. Actual:       `abcd` 
                2. Predicted:    `abc-` 
                3. (<Deleted Phoneme, DELETION>, count+=1)
            3. Insertion
                1. Actual:       `abc-`
                2. Predicted:    `abcd`
                3. (<INSERTION, Inserted Phoneme>, count+=1)
            4. Substitution
                1. Predicted:    `abc`
                2. Actual:       `abd`
    """
    confusion_counts = Counter()

    # Step 1) For each transcription/audio file given, count the actual/predicted alignments
    for true_transcrip, pred_transcrip in true_pred_pairs:
        # Step 2) Extract the transcription and align them
        alignments = pairwise2.align.globalxx(true_transcrip, pred_transcrip)

        # Step 3) Compare the top 3 transcriptions if available and add to confusion matrix
        for align in alignments[:3]:
            # Step 4) Normalize the alignments based on the number of alignments used
            normalization = len(alignments[:3])
            updateAmt = 1 / normalization

            for actual_token, pred_token in zip(align.seqA, align.seqB):
                # Case 1) Insertion
                if actual_token == "-":
                    confusion_counts[("INS", pred_token)] += updateAmt
                # Case 2) Deletion
                elif pred_token == "-":
                    confusion_counts[(actual_token, "DEL")] += updateAmt
                # Case 3) Substitution
                else:
                    confusion_counts[(actual_token, pred_token)] += updateAmt

    # Step 5) Visualize alignments as a confusion matrix
    # Step 5A) Prep the lexicon with edit markings
    lexicon = format_lexicon(lexicon)
    lexicon_matrix_lookup = {
        token: i for i, token in enumerate(lexicon)
    }

    # Step 5B) Populate confusion matrix
    matrix = create_base_confusion_matrix(lexicon=lexicon)

    # X axis is actual, Y axis is predicted
    for (actual, predicted), count in confusion_counts.items():
        matrix[lexicon_matrix_lookup[actual],
               lexicon_matrix_lookup[predicted]] = count
    return matrix


def visualize_confusion_matrix(confusion_matrix: np.ndarray, lexicon: list[str] = BASE_LEXICON, title:str="") -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, xticklabels=format_lexicon(lexicon), yticklabels=format_lexicon(lexicon))

    plt.title(f"Confusion Matrix: Aligned Phonemes {title}" )
    plt.xlabel("Actual Phonemes")
    plt.ylabel("Predicted Phonemes")
    
    plt.show()
