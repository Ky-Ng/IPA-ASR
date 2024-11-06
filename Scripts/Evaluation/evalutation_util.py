def remove_diacritics(transcription: list[str]) -> list[str]:
    """
    Remove vowel lengthening marker 
    TODO if there are other diacritics to be removed in the future
    """
    no_vowel_lengthening_marker = [token.replace("ː", "") for token in transcription]
    
    return no_vowel_lengthening_marker
    