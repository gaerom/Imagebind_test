from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# 5개 확인 

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

# text_list=["dog growling", "male singing", "people babbling", "police car (siren)", "yodelling"]
# image_paths=["./modalities/dog growling.png", "./modalities/male singing.png", "./modalities/people babbling.png", "./modalities/police car (siren).png", "./modalities/yodelling.png"]
# audio_paths=["./modalities/dog growling_32720.wav", "./modalities/male singing_27393.wav", "./modalities/people babbling_88641.wav", "./modalities/police car (siren)_95560.wav", "./modalities/yodelling_124652.wav"]


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)


# similarity
vt_sim = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T
at_sim = embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T
va_sim = embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T


# dot product에 softmax를 한 결과
vt_soft = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
at_soft = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
va_soft = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1)


print("Vision x Text similarity: ")
print(vt_sim)
print("Audio x Text similarity: ")
print(at_sim)
print("Vision x Audio similarity:")
print(va_sim)



# print("Vision x Text: ")
# print(vt_soft)
# print("Audio x Text: ")
# print(at_soft)
# print("Vision x Audio:")
# print(va_soft)

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])