from utils import predict_stage

file = "test_audio.wav"
result = predict_stage(file)
print("🎧 Predicted Sleep Stage:", result)
