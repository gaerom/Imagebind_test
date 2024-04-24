import torch
from torch.utils.data import Dataset, DataLoader
from torch.linalg import norm
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data

class MultimodalDataset(Dataset):
    def __init__(self, audio_dir, image_dir, text_file, transform=None):
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.text_file = text_file
        self.transform = transform
        
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.image_files = [f.replace('.wav', '.png') for f in self.audio_files]
        
        with open(text_file, 'r') as f:
            self.texts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_filename = self.audio_files[idx]
        image_filename = self.image_files[idx]
        text = self.texts[idx]  
        
        audio_path = os.path.join(self.audio_dir, audio_filename)
        image_path = os.path.join(self.image_dir, image_filename)
        
        audio_data = data.load_and_transform_audio_data([audio_path], device)
        image_data = data.load_and_transform_vision_data([image_path], device)
        text_data = data.load_and_transform_text([text], device)

        if image_data.ndim > 4:  # image_data can be a 5D tensor [1, 3, 1, H, W]
            image_data = image_data.squeeze(2)
        
        return audio_data, image_data, text_data, (audio_filename, image_filename, text)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True).to(device)
model.eval()

# audio_dir = './modalities/audios'
# image_dir = './modalities/frames_test' # for test
# text_file = './modalities/labels.txt' # for test

# 실제 similarity 계산에 사용했던 경로
# audio_dir = '/mnt/storage1/trainvideo_10_audios'
# image_dir = './modalities/frames_'
# text_file = './modalities/transformed_texts.txt'

# 9개 선정한 동영상에 대해 분석
audio_dir = './modalities/t-SNE/audios'
image_dir = './modalities/t-SNE/frames'
text_file =  './modalities/t-SNE/label.txt'

dataset = MultimodalDataset(audio_dir, image_dir, text_file)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 각 embedding 크기를 저장할 list
audio_mags = []
image_mags = []
text_mags = []


vt_sims = []
at_sims = []
va_sims = []

filenames_list = [] 

for audio_batch, image_batch, text_batch, filenames in dataset:
    with torch.no_grad():
        
        inputs = {
            ModalityType.AUDIO: audio_batch,
            ModalityType.VISION: image_batch,
            ModalityType.TEXT: text_batch
        }
        
        embeddings = model(inputs)
        
        # dot product 수행 결과
        # vt_dot = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T
        # at_dot = embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T
        va_dot = embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T


        # 같은 결과끼리 확인해보기
        # vision_dot = embeddings[ModalityType.VISION] @ embeddings[ModalityType.VISION].T
        # text_dot = embeddings[ModalityType.TEXT] @ embeddings[ModalityType.TEXT].T
        # audio_dot = embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.AUDIO].T
        

        # print(f'vision: {vision_dot}')
        # print(f'text: {text_dot}') 
        # print(f'audio: {audio_dot}')
        # print('----------------------------------------------------------\n')



        # vector magnitude 계산 
        # audio_mag = norm(embeddings[ModalityType.AUDIO], dim=1)
        # image_mag = norm(embeddings[ModalityType.VISION], dim=1)
        # text_mag = norm(embeddings[ModalityType.TEXT], dim=1)


        # 위 결과와 똑같음
        audio_mag = torch.sqrt(torch.sum(embeddings[ModalityType.AUDIO]**2, dim=1))
        image_mag = torch.sqrt(torch.sum(embeddings[ModalityType.VISION]**2, dim=1))
        text_mag = torch.sqrt(torch.sum(embeddings[ModalityType.TEXT]**2, dim=1))

        audio_mags.append(audio_mag.cpu().numpy())
        image_mags.append(image_mag.cpu().numpy())
        text_mags.append(text_mag.cpu().numpy())


        # cosine similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6) # 1e-8
        # vt_sim = cos(embeddings[ModalityType.VISION], embeddings[ModalityType.TEXT])
        # at_sim = cos(embeddings[ModalityType.AUDIO], embeddings[ModalityType.TEXT])
        va_sim = cos(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO])
        
        # GPU tensor를 CPU로 이동
        # vt_sims.append(vt_sim.cpu().numpy())
        # at_sims.append(at_sim.cpu().numpy())
        va_sims.append(va_sim.cpu().numpy())
        
        for _ in range(len(va_dot)):
            filenames_list.append(filenames)
        
        
        print(f'Files: {filenames}') # modality pair들이 알맞게 들어가는지 확인 가능
        # print('Vision x Text similarity: ', vt_sim)
        # print('Audio x Text similarity: ', at_sim)
        print('Vision x Audio similarity: ', va_sim)
        print('----------------------------------------------------------\n')
        
        # print('Audio Magnitude: ', audio_mag)
        # print('Image Magnitude: ', image_mag)
        # print('Text Magnitude: ', text_mag)

        # print('----------------------------------------------------------\n')
        
# vt_sims = np.concatenate(vt_sims, axis=None)
# at_sims = np.concatenate(at_sims, axis=None)
# va_sims = np.concatenate(va_sims, axis=None)       


# # median 계산
# vt_median = np.median(vt_sims)
# at_median = np.median(at_sims)
# va_median = np.median(va_sims)


# with open('/mnt/storage1/vggsoundsync/below_median_test_.txt', 'w') as file: # 경로 수정
#     for i, (vt, at, va) in enumerate(zip(vt_sims, at_sims, va_sims)):
#         if vt < vt_median and at < at_median and va < va_median: # 모두 넘지 못하는 경우
#             audio_filename = filenames_list[i][0] 
#             file.write(f"{audio_filename}\n")
#             # print(audio_filename)