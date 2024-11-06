# Translating from TIMIT to IPA


# Lookup table to translate from TIMIT Transcriptions to IPA
# for our MVP, we're looking to distinguish only
# Aspirated vs. Unaspirated voicless stops

'''
Maps <timit_symbol, ipa_symbol> if there is 
a discrepancy between the two transcriptions

If the transcriptions are the same, no pair is added
'''
TIMIT_IPA_LOOKUP = {
    # Aspirated vs. Unaspirated Stops
    'tcl': 't',
    't': 'tʰ',

    'kcl': 'k',
    'k': 'kʰ',

    'pcl': 'p',
    'p': 'pʰ',

    # Voiced Closures (don't differentiate released/unreleased)
    'bcl': 'b',
    'dcl': 'd',
    'gcl': 'g',

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
TIMIT_WIDE_TRANSCRIPTION_LOOKUP = {
    # Voiced Closures (don't differentiate released/unreleased)
    'b': '',
    'd': '',
    'g': '',

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

"""
Maps each Release to its closure
- Used for preprocessing TIMIT transcriptions which 
    make it seem like there's a release without a closure
"""
RELEASE_CLOSURE_PAIRS = {
    'b': 'bcl',
    'p': 'pcl',

    'd': 'dcl',
    't': 'tcl',

    'g': 'gcl',
    'k': 'kcl'
}

UNASPIRATED_ASPIRATED_PAIRS = {
    't': 'tʰ',
    'k': 'kʰ',
    'p': 'pʰ',
}

"""
Lookup table of how to convert each TIMIT transcription to IPA
"""
FULL_TIMIT_LOOKUP = {**TIMIT_IPA_LOOKUP, **TIMIT_WIDE_TRANSCRIPTION_LOOKUP}


def enforceClosureForReleases(origTimitTrans) -> list[str]:
    """
    Ensures there is a closure for a stop if the release is present
    """
    consecPairs = []
    for i in range(len(origTimitTrans)-1):
        consecPairs.append((origTimitTrans[i], origTimitTrans[i+1]))

    # Add back the last element as a left child to not lose any items in 
    consecPairs.append((origTimitTrans[-1], ""))
    ret = []

    # Loop through all consec pairs where we may have (l != CLosure) and (r = RELease)
    for l, r in consecPairs:
        # Step 1) Reconstruct Base Array by adding left child
        ret.append(l)

        # Case) (l != CLosure) and (r = RELease), add the closure in
        if r in RELEASE_CLOSURE_PAIRS and l != RELEASE_CLOSURE_PAIRS[r]:
            ret.append(RELEASE_CLOSURE_PAIRS[r])

    return ret

def mergeAspiration(finishTrans) -> list[str]:
    """
    If there is a CLO + REL, the lookup will transcribe them as xxʰ instead of just xʰ
    This script removes the xʰ in favor of simplicity and not introducing new vocabulary items

    TODO: add `xʰ` as the vocab item, currently remove xʰ for simplicity
    """
    consecPairs = [("", finishTrans[0])]
    for i in range(len(finishTrans)-1):
        consecPairs.append((finishTrans[i], finishTrans[i+1]))

    ret = []

    # Loop through all consec pairs where we may have (l != CLosure) and (r = RELease)
    for l, r in consecPairs:
        # Case 1)  pair of (x, xʰ), skip xʰ
        if l in UNASPIRATED_ASPIRATED_PAIRS and r == UNASPIRATED_ASPIRATED_PAIRS[l]:
            #TODO select the xʰ in the future
            continue
        
        # Case 2) Add normally
        ret.append(r)
    return ret


def getTimitLookup(timitTranscription: list[str]) -> list[str]:
    """
    Converts TIMIT transcription to desired phoneme representation
    without pre/post processing
    """
    ret = []
    for letter in timitTranscription:
        if letter in FULL_TIMIT_LOOKUP:
            ret.append(FULL_TIMIT_LOOKUP[letter])
        else:
            ret.append(letter)
    return ret


def getTimitToIPA(timit: list[str]) -> list[str]:
    # Strip whitespace and lowercase input
    preprocessedTimit = [token.strip().lower()
                         for token in enforceClosureForReleases(timit)]

    # Convert each token to IPA
    rawIpaLookup = getTimitLookup(preprocessedTimit)

    # Remove Apsiration TODO add back when ready to handle new vocabulary items
    removeAspiration = mergeAspiration(rawIpaLookup)
    
    # Remove empty spaces
    removeEmpty = [token for token in removeAspiration if len(token) != 0]
    
    return removeEmpty


def compareTranscriptions(trans1: list[str], trans2: list[str]) -> None:
    """
    Used for debugging comparisons of two transcriptions side by side
    """
    idx = 0
    while idx < len(trans1) or idx < len(trans2):
        l = r = ""
        if idx < len(trans1):
            l = trans1[idx]
        if idx < len(trans2):
            r = trans2[idx]

        print(f"{l:3} | {r:3}")
        idx += 1
