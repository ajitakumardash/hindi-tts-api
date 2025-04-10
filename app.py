from flask import Flask, request, jsonify, send_file
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import uuid
import os

app = Flask(__name__)

# ✅ Try loading Kokoro TTS model with debug info
try:
    pipeline = KPipeline(lang_code='h')  # Hindi TTS pipeline
    print("✅ Kokoro pipeline loaded successfully")
except Exception as e:
    print("❌ Error loading Kokoro pipeline:", e)
    raise e  # Force app to stop if loading fails

# 📁 Output directory for audio files
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔉 POST /synthesize endpoint
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get("text", "")
    voice = data.get("voice", "hf_alpha")
    speed = data.get("speed", 1.0)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        print("🎤 Synthesizing:", text[:100])
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'[।.!?\n]+')
        audio_chunks = [audio for _, _, audio in generator]

        if not audio_chunks:
            return jsonify({"error": "Failed to generate audio"}), 500

        final_audio = np.concatenate(audio_chunks)
        filename = f"{uuid.uuid4().hex}.wav"
        file_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(file_path, final_audio, 24000)

        print(f"✅ Audio generated: {file_path}")
        return send_file(file_path, mimetype='audio/wav')

    except Exception as e:
        print("❌ Synthesis error:", str(e))
        return jsonify({"error": str(e)}), 500

# 🌐 GET / - health check
@app.route('/')
def home():
    return "✅ Hindi TTS API is running!"

# 🚀 App startup
if __name__ == '__main__':
    print("✅ Flask app starting...")
    app.run(host='0.0.0.0', port=5000)
