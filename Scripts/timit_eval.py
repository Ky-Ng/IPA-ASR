from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from TIMIT_Helpers.timit_lookup import getTimitToIPA, compareTranscriptions
import librosa
import torch

# Load in Model
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
timit = load_dataset('timit_asr', data_dir="TIMIT-Database/TIMIT")

# Constants
TIMIT_EXAMPLE = 1
TARGET_SR = 16000
audio_path = timit['train']['file'][TIMIT_EXAMPLE] # ex: Thick glue oozed ...

# Resample Audio
waveform, original_sr = librosa.load(audio_path)
resampled_audio = librosa.resample(
    waveform, orig_sr=original_sr, target_sr=TARGET_SR)

# Process Audio
input_values = processor(resampled_audio, return_tensors="pt",
                         sampling_rate=TARGET_SR).input_values
input_values = torch.reshape(input_values, (1, -1))

with torch.no_grad():
    logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

# TIMIT cleaned up transcription
orig_timit_transcription = timit['train']['phonetic_detail'][TIMIT_EXAMPLE]['utterance']
ipa_timit_transcription = getTimitToIPA(orig_timit_transcription)

print(audio_path)
print(timit['train']['text'][TIMIT_EXAMPLE])
compareTranscriptions(ipa_timit_transcription, transcription[0].split(" "))


