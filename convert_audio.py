from pydub import AudioSegment

source_path = "Snoring Dataset/0/0_6.wav"
output_path = "test_audio.wav"

# Convert the dataset audio to real WAV format
sound = AudioSegment.from_file(source_path)
sound.export(output_path, format="wav")

print("✅ Converted to real WAV file!")
