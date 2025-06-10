from deepface import DeepFace
result = DeepFace.analyze(
    img_path = "test.jpg",
    actions = ['emotion']
)

print("Analysis Result:")
print(result)