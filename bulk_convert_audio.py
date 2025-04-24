from pydub import AudioSegment
import os

# Convert both folders: 0 (Awake) and 1 (DeepSleep)
for label in ["0", "1"]:
    input_folder = f"Snoring Dataset/{label}"
    output_folder = f"Converted/{label}"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            source_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            sound = AudioSegment.from_file(source_path)
            sound.export(output_path, format="wav")
            print(f"✅ Converted: {label}/{filename}")
