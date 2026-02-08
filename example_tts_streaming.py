import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import time
# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
# wav = model.generate(text)
# ta.save("non_streaming.wav", wav, model.sr)
start_time = time.time()
wav = model.generate_streaming(text)
end_time = time.time()
print(f"Total generation time: {end_time - start_time:.3f}s")
print(f"RTF: {(end_time - start_time) / (wav.shape[-1] / model.sr):.3f}")
print(f"Total audio duration: {wav.shape[-1] / model.sr:.3f}s")
ta.save("streaming.wav", wav, model.sr)
