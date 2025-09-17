import os
import json
import numpy as np
import subprocess
from scipy.io.wavfile import write
from kokoro_onnx import Kokoro
from misaki import espeak
from misaki.espeak import EspeakG2P

# --- SCRIPT CONFIGURATION ---
INPUT_TEXT = """La serie The Swarm, un eco-thriller sobre el cambio climático, ha sido estrenada en HBO Max. Es una adaptación del bestseller de Frank Schatzing y consta de 8 episodios en su primera temporada. La trama comienza con eventos extraños en las profundidades del océano, donde las criaturas se vuelven más hostiles. En Canadá, las ballenas se han vuelto violentas, atacando barcos que naufragan. Un patógeno mortal se propaga, envenenando el marisco. Un grupo de científicos estudia estos casos, descubriendo que son ataques coordinados con un misterioso punto común, declarando la guerra a la humanidad. La serie tiene un trasfondo ecológico importante y mantendrá en vilo a los espectadores."""
VOICE_CHOICE = 'Dora'
SPEED = 1.0

# --- CHUNKING & MERGING PARAMETERS ---
MIN_CHUNK_LENGTH = 60
MAX_CHUNK_LENGTH = 100
MIN_LAST_PHRASE_LENGTH = 3  # Prevents cutting on abbreviations like "Sr."
SILENCE_MS = 150			# Silence to add between merged audio chunks

# --- FILE PATHS & NAMES ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODEL_DIR, "voices-v1.0.bin")

WAV_OUTPUT_FILENAME = "direct_audio.wav"
MP3_OUTPUT_FILENAME = "final_audio.mp3"
TIMESTAMPS_FILENAME = "timestamps.json"

# --- MODEL MAPPINGS ---
VOICES = {'Dora': 'ef_dora', 'Alex': 'em_alex', 'Santa': 'em_santa'}


def split_text_into_chunks(text, min_len, max_len, min_last_word):
	"""
	Intelligently splits a block of text into chunks based on length constraints.
	It prioritizes splitting at the end of sentences.
	"""
	sentences = text.replace('.\n', '. ').replace('\n', ' ').split('. ')
	chunks = []
	current_chunk = ""

	for i, sentence in enumerate(sentences):
		if not sentence.strip():
			continue

		original_ending = ". " if i < len(sentences) - 1 else "."
		sentence_with_punct = sentence + (original_ending if not text.endswith(sentence) else "")

		if current_chunk and len(current_chunk) + len(sentence_with_punct) > max_len:
			if len(current_chunk) < min_len:
				current_chunk += sentence_with_punct
			else:
				last_word = current_chunk.strip().split(' ')[-1]
				if len(last_word) < min_last_word and last_word.endswith('.'):
					current_chunk += sentence_with_punct
				else:
					chunks.append(current_chunk.strip())
					current_chunk = sentence_with_punct
		else:
			current_chunk += sentence_with_punct

	if current_chunk.strip():
		chunks.append(current_chunk.strip())
		
	return chunks


def generate_audio(text, kokoro_model, g2p_model, voice_id, speed):
	"""
	Generates audio samples from a text string using the loaded Kokoro model.
	"""
	phonemes, _ = g2p_model(text)
	samples, sample_rate = kokoro_model.create(phonemes, voice_id, speed, is_phonemes=True)
	return samples, sample_rate


def main():
	"""
	Main execution pipeline for generating calibrated TTS audio and timestamps.
	"""
	if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
		raise FileNotFoundError("Model files not found. Please check the 'model' directory.")

	print("Initializing TTS models...")
	espeak.EspeakFallback(british=False)
	g2p = EspeakG2P(language="es")

	# It will attempt to use CUDA first and fall back to the CPU if it fails.
	cuda_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
	kokoro = Kokoro(MODEL_PATH, VOICES_PATH, providers=cuda_providers)
	
	selected_voice_id = VOICES.get(VOICE_CHOICE)
	if not selected_voice_id:
		raise ValueError(f"Invalid voice choice: '{VOICE_CHOICE}'")
	print("Models loaded successfully (Attempting to use CUDA).")

	# --- STEP 1: Generate the direct, high-quality audio file ---
	print(f"\n--- STEP 1: Generating direct audio -> '{WAV_OUTPUT_FILENAME}' ---")
	direct_samples, sample_rate = generate_audio(INPUT_TEXT, kokoro, g2p, selected_voice_id, SPEED)
	write(WAV_OUTPUT_FILENAME, sample_rate, direct_samples)
	print("Direct audio generated.")

	# --- STEP 2: Split the input text into manageable chunks ---
	print("\n--- STEP 2: Splitting text into phrases ---")
	phrases = split_text_into_chunks(INPUT_TEXT, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH, MIN_LAST_PHRASE_LENGTH)
	for i, p in enumerate(phrases):
		print(f"  Phrase {i+1} ({len(p)} chars): {p}")

	# --- STEP 3 & 4: Generate each chunk and create baseline timestamps ---
	print(f"\n--- STEP 3 & 4: Generating {len(phrases)} individual chunks for timing... ---")
	audio_chunks = []
	base_timestamps = []
	current_time_ms = 0
	for i, phrase in enumerate(phrases):
		samples, rate = generate_audio(phrase, kokoro, g2p, selected_voice_id, SPEED)
		audio_chunks.append(samples)
		duration_ms = (len(samples) / rate) * 1000
		base_timestamps.append({
			"phrase_num": i,
			"phrase": phrase,
			"start_ms": int(current_time_ms),
			"end_ms": int(current_time_ms + duration_ms)
		})
		current_time_ms += duration_ms + SILENCE_MS
		print(f"  Generated phrase {i+1} ({duration_ms:.0f} ms).")

	# --- STEP 5: Generate the merged audio in-memory for duration calculation ---
	print("\n--- STEP 5: Merging audio chunks in-memory ---")
	silence_samples = np.zeros(int(SILENCE_MS / 1000 * sample_rate), dtype=np.int16)
	final_audio_list = []
	for i, chunk in enumerate(audio_chunks):
		final_audio_list.append(chunk)
		if i < len(audio_chunks) - 1:
			final_audio_list.append(silence_samples)
	merged_samples = np.concatenate(final_audio_list)
	print("In-memory merged audio created.")

	# --- STEP 6: Calibrate timestamps using the scaling factor ---
	print("\n--- STEP 6: Calibrating timestamps to match direct audio ---")
	duration_direct_sec = len(direct_samples) / sample_rate
	duration_merged_sec = len(merged_samples) / sample_rate
	print(f"  Direct audio duration: {duration_direct_sec:.3f} seconds")
	print(f"  Merged audio duration: {duration_merged_sec:.3f} seconds")

	scaling_factor = duration_direct_sec / duration_merged_sec if duration_merged_sec > 0 else 1.0
	print(f"  Calculated scaling factor: {scaling_factor:.5f}")

	corrected_timestamps = []
	for item in base_timestamps:
		corrected_start_ms = item["start_ms"] * scaling_factor
		corrected_end_ms = item["end_ms"] * scaling_factor
		corrected_timestamps.append({
			"phrase_num": item["phrase_num"],
			"phrase": item["phrase"],
			"start_ms": int(corrected_start_ms),
			"end_ms": int(corrected_end_ms)
		})

	# --- STEP 7: Save the calibrated timestamps to a JSON file ---
	print(f"\n--- STEP 7: Saving calibrated timestamps -> '{TIMESTAMPS_FILENAME}' ---")
	with open(TIMESTAMPS_FILENAME, 'w', encoding='utf-8') as f:
		json.dump(corrected_timestamps, f, indent=2, ensure_ascii=False)
	print("Calibrated JSON file saved.")

	# --- STEP 8: Convert the final WAV audio to MP3 and clean up ---
	print(f"\n--- STEP 8: Converting to MP3 -> '{MP3_OUTPUT_FILENAME}' ---")
	command = [
		'ffmpeg',
		'-i', WAV_OUTPUT_FILENAME,
		'-y',
		'-loglevel', 'error',
		MP3_OUTPUT_FILENAME
	]
	try:
		subprocess.run(command, check=True)
		print(f"MP3 file saved successfully.")
		os.remove(WAV_OUTPUT_FILENAME)
		print(f"Temporary file '{WAV_OUTPUT_FILENAME}' removed.")
	except FileNotFoundError:
		print("\nERROR: 'ffmpeg' command not found. Ensure it's installed and in your system's PATH.")
	except subprocess.CalledProcessError as e:
		print(f"\nERROR: ffmpeg failed during conversion. Details: {e}")

	print(f"\n\nProcess complete! Final audio is in '{MP3_OUTPUT_FILENAME}' and timestamps are in '{TIMESTAMPS_FILENAME}'.")


if __name__ == "__main__":
	main()