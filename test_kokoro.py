
from kokoro_onnx import Kokoro
import soundfile as sf

MODEL = "/home/richard/jarvis/kokoro/kokoro-v1.0.onnx"
VOICES = "/home/richard/jarvis/kokoro/voices-v1.0.bin"

print("Loading Kokoro...")

kokoro = Kokoro(MODEL, VOICES)

print("Generating test speech...")

samples, sample_rate = kokoro.create(
    "Good evening. JARVIS is online and ready.",
    voice="bm_george",
    speed=0.95,
    lang="en-gb"
)

sf.write("kokoro_test.wav", samples, sample_rate)

print("SUCCESS: Audio saved to kokoro_test.wav")
