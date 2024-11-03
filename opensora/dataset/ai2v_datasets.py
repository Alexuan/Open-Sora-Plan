import json
import os, io, csv, math, random
import cv2
import scipy.io
from scipy.io import wavfile
import numpy as np
import torchvision
import librosa
from einops import rearrange
from decord import VideoReader
from os.path import join as opj

import torch
import torchvision.transforms as transforms
import torchaudio
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from .t2v_datasets import T2V_dataset

def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

class AI2V_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.image_data = args.image_data
        self.video_data = args.video_data
        self.audio_data = args.audio_data
        self.num_frames = args.num_frames
        self.cond_image_num = args.cond_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.v_decoder = DecordInit()
        self.transform_audio = torchaudio.transforms.Spectrogram(n_fft=1000)

        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
            self.aud_cap_list = self.get_aud_cap_list()
            self.img_cap_list = self.get_img_cap_list()
        else:
            self.img_cap_list = self.get_img_cap_list()
        
        
    def __getitem__(self, idx):
        try:
            # video_data, image_data, audio_data = {}, {}, {}
            if self.num_frames != 1:
                video_data, audio_data = self.get_video_audio(idx)
                image_data = self.get_image(idx)
            else:
                image_data = self.get_image(idx)  # 1 frame video as image
            file_name = os.path.basename(self.vid_cap_list[idx]['path'])[:-4]
            return_dict = dict(video_data=video_data, 
                               image_data=image_data, 
                               audio_data=audio_data, 
                               file_name=file_name)
            return return_dict
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_video_audio(self, idx):
        # video = random.choice([random_video_noise(65, 3, 720, 360) * 255, random_video_noise(65, 3, 1024, 1024), random_video_noise(65, 3, 360, 720)])
        # # print('random shape', video.shape)
        # input_ids = torch.ones(1, 120).to(torch.long).squeeze(0)
        # cond_mask = torch.cat([torch.ones(1, 60).to(torch.long), torch.ones(1, 60).to(torch.long)], dim=1).squeeze(0)
        
        # TODO (Xuan): look into input_ids & cond_mask
        video_path = self.vid_cap_list[idx]['path']
        # NOTE (Xuan): temporaly comment out the frame_idx
        # frame_idx = self.vid_cap_list[idx]['frame_idx']
        # video = self.decord_read(video_path, frame_idx)
        video, time_begin, time_end = self.decord_read(video_path)
        video = self.transform(video)  # T C H W -> T C H W
        # video = torch.rand(65, 3, 512, 512)
        video = video.transpose(0, 1)  # T C H W -> C T H W

        audio_path = self.aud_cap_list[idx]['path']
        audio = self.decord_read_audio(audio_path, time_begin, time_end)

        text = self.vid_cap_list[idx]['cap']
        text = text_preprocessing(text)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']

        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask), dict(audio=audio)
        # return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)
        

    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames-1, self.cond_image_num, dtype=int)
        assert self.num_frames >= self.cond_image_num
        image = [video_data['video'][:, i:i+1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data['input_ids'].repeat(self.cond_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.cond_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, idx):
        idx = idx % len(self.img_cap_list)  # out of range
        image_data = self.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        image = [Image.open(i['path']).convert('RGB') for i in image_data] # num_img [h, w, c]
        image = [torch.from_numpy(np.array(i)) for i in image] # num_img [h, w, c]
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image] # num_img [1 c h w]
        image = [self.transform(i) for i in image]  # num_img [1 C H W] -> num_img [1 C H W]
        # image = [torch.rand(1, 3, 512, 512) for i in image_data]
        image = [i.transpose(0, 1) for i in image]  # num_img [1 C H W] -> num_img [C 1 H W]

        caps = [i['cap'] for i in image_data]
        text = [text_preprocessing(cap) for cap in caps]
        input_ids, cond_mask = [], []
        for t in text:
            text_tokens_and_mask = self.tokenizer(
                t,
                max_length=self.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids.append(text_tokens_and_mask['input_ids'])
            cond_mask.append(text_tokens_and_mask['attention_mask'])
        input_ids = torch.cat(input_ids)  # self.use_image_num, l
        cond_mask = torch.cat(cond_mask)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def tv_read(self, path, frame_idx=None):
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video = vframes[frame_indice]  # (T, C, H, W)

        return video
    
    def decord_read(self, path, frame_idx=None):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            # start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)

        fps = get_video_fps(path)
        time_start = float(start_frame_ind / fps)
        time_end = float(end_frame_ind / fps)
        return video_data, time_start, time_end


    def decord_read_audio(self, path, time_start, time_end, frame_idx=None):
        sr, audio = wavfile.read(path)
        if sr == 44100 and audio.shape[1] == 2:
            audio = librosa.resample(np.mean(audio, axis=1,), 
                                     orig_sr=44100, 
                                     target_sr=16000,
                                     res_type="scipy")
            sr = 16000
        frame_start = int(sr * time_start)
        frame_end = int(sr * time_end)
        audio_chunk = audio[frame_start:frame_end]
        audio_chunk = torch.from_numpy(audio_chunk)

        # spec_chunk = self.transform_audio(audio_chunk) # B * F * T
        return audio_chunk


    def get_vid_cap_list(self):
        vid_cap_lists = []
        with open(self.video_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
            # print(folder_anno)
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                vid_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(vid_cap_list))):
                path = opj(folder, vid_cap_list[i]['path'])
                if os.path.exists(path.replace('.mp4', '_resize_1080p.mp4')):
                    path = path.replace('.mp4', '_resize_1080p.mp4')
                vid_cap_list[i]['path'] = path
            vid_cap_lists += vid_cap_list
        return vid_cap_lists

    def get_img_cap_list(self):
        use_image_num = self.cond_image_num if self.cond_image_num != 0 else 1
        img_cap_lists = []
        with open(self.image_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                img_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(img_cap_list))):
                img_cap_list[i]['path'] = opj(folder, img_cap_list[i]['path'])
            img_cap_lists += img_cap_list
        img_cap_lists = [img_cap_lists[i: i+use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
        return img_cap_lists[:-1]  # drop last to avoid error length

    def get_aud_cap_list(self):
        aud_cap_lists = []
        with open(self.audio_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                aud_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(aud_cap_list))):
                aud_cap_list[i]['path'] = opj(folder, aud_cap_list[i]['path'])
            aud_cap_lists += aud_cap_list
        return aud_cap_lists
