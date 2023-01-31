import whisper

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = model.load_audio("AudiosToConvert/ThomasAudio.mp3")
audio = model.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = model.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")
