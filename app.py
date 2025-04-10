from flask import Flask, request, jsonify, send_file
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import uuid
import os

app = Flask(__name__)
pipeline = KPipeline(lang_code='h')  # Hindi TTS pipeline

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get("text", "")
    voice = data.get("voice", "hf_alpha")
    speed = data.get("speed", 1.0)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'[।.!?\n]+')
        audio_chunks = [audio for _, _, audio in generator]

        if not audio_chunks:
            return jsonify({"error": "Failed to generate audio"}), 500

        final_audio = np.concatenate(audio_chunks)
        filename = f"{uuid.uuid4().hex}.wav"
        file_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(file_path, final_audio, 24000)

        return send_file(file_path, mimetype='audio/wav')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "✅ Hindi TTS API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
