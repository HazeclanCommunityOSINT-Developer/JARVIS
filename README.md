# JARVIS

The Python virtual environment is local and is not tracked in Git. Install the
runtime dependencies in your own environment before running the application.

Kokoro speech model assets (`kokoro/*.onnx` and `kokoro/*.bin`) are also kept
outside Git because of their size. Place the model and voice files in
`~/jarvis/kokoro/` to enable Kokoro; JARVIS can fall back to Piper when those
files are unavailable. `kokoro_test.wav` is generated output and is not tracked.
# JARVIS
