# Lookup table to translate from TIMIT Transcriptions to IPA

# for our MVP, we're looking to distinguish only
# Aspirated vs. Unaspirated voicless stops

'''
Maps <timit_symbol, ipa_symbol> if there is 
a discrepancy between the two transcriptions

If the transcriptions are the same, no pair is added
'''
timit_ipa_lookup = {
    # Aspirated vs. Unaspirated Stops
    'tcl': 't',
    't': 'tʰ',

    'kcl': 'k',
    'k': 'kʰ',

    'pcl': 'p',
    'p': 'pʰ',

    # Flaps
    "dx": 'ɾ',

    # Glottal Stop
    'q': 'ʔ',

    # Affricates
    'jh': 'dʒ',
    'ch': 'tʃ',

    # Fricatives
    'sh': 'ʃ',
    'zh': 'ʒ',
    'th': 'θ',
    'dh': 'ð',

    # Nasals
    'ng': 'ŋ',

    # Glides
    'y': 'j',
    'hh': 'h',

    # Front Vowels
    'iy': 'i',
    'ih': 'ɪ',
    'eh': 'ɛ',
    'ae': 'æ',
    
    # Central Vowels
    'ah': 'ʌ',
    'ax': 'ə',
    'axr': 'ɚ',
    'er': 'ɛ˞',

    # Back Vowels
    'uw': 'u',
    'uh': 'ʊ',
    'ao': 'ɔ',
    'aa': 'ɑ',
    
    # Dipthongs
    'ey': 'eɪ',
    'ay': 'aɪ',
    'aw': 'aʊ',
    'oy': 'ɔɪ',
    'ow': 'oʊ',
}

""" 
some of the TIMIT translations are more narrow
or have transcriptions that may not be
as useful for cross-linguistic generalization
currently and are thus mapped onto a wider IPA transcription
"""

timit_wider_transcription_lookup = {
    # Nasals
    'em': 'm',  # syllabic m
    'en': 'n',  # syllabic n
    'eng': 'ŋ',  # syallbic ŋ

    'nx': 'n',  # Voiced nasal flap, prefer n in transcription?

    # Glides
    'hv': 'h',  # Check on phonetics def of stress
    'el': 'l',  # dark l (IPA: ɫ)

    # Vowels
    'ux': 'u',  # fronted /u/ usually in alveolar context
    'ix': 'ɪ',  # Check if this is a stressed allophone
    'ax-h': 'ə',  # devoiced shwa (ə̥)

    # Other symbols (remove stress markers and breaks)
    'pau': '',
    'epi': '',
    'h#': '',
    '1': '',
    '2': '',
}
