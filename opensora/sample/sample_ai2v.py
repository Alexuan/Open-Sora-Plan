import math
import os
import torch
import argparse
import torchvision
import cv2
import numpy as np

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText

import os, sys

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.diffusion.latte.modeling_latte_a2v import LatteA2V
from opensora.models.text_encoder import get_text_enc
from opensora.models.audio_encoder import get_audio_warpper
from opensora.utils.utils import save_video_grid
from opensora.dataset import getdataset
from opensora.utils.dataset_utils import CollateAI2V

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen_ai2v import VideoGenPipelineAI2V

import imageio


def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE (Xuan): try add .eval()
    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir').to(device, dtype=torch.float16)
    if args.ae_ckpt_dir:
        ae_ckpt = torch.load(args.ae_ckpt_dir)
        ae_ckpt_sd_update = dict()
        for k, v in ae_ckpt["state_dict"].items():
            k_update = "vae." + k
            ae_ckpt_sd_update[k_update] = v
        vae.load_state_dict(ae_ckpt_sd_update)
    
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        
    # Load model:
    transformer_model = LatteA2V.from_pretrained(
        "121x80x80_speechlm_bs8_lr2e-5_1img_sr1", 
        subfolder="checkpoint-98000/model", 
        cache_dir="cache_dir",
        torch_dtype=torch.float16).to(device)
    transformer_model.force_images = args.force_images

    kwargs = {'load_in_8bit': args.enable_8bit_t5, 'torch_dtype': torch.float16, 'low_cpu_mem_usage': False} # NOTE (Xuan): accelerator error
    audio_encoder = get_audio_warpper(args.audio_encoder_name)(args, **kwargs).eval()
    audio_encoder.dtype = torch.float16
    audio_encoder.device = device

    video_length, image_size = transformer_model.config.video_length, int(args.version.split('x')[1])
    latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    vae.latent_size = latent_size
    if args.force_images:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    # set eval mode
    transformer_model.eval()
    vae.eval()
    audio_encoder.eval()

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    print('videogen_pipeline', device)
    videogen_pipeline = VideoGenPipelineAI2V(vae=vae,
                                             audio_encoder=audio_encoder,
                                             scheduler=scheduler,
                                             transformer=transformer_model).to(device=device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)
        os.makedirs(os.path.join(args.save_img_path, "pred"), exist_ok=True)
        os.makedirs(os.path.join(args.save_img_path, "groundtruth"), exist_ok=True)
        os.makedirs(os.path.join(args.save_img_path, "pred_groundtruth"), exist_ok=True)

    video_grids = []
    test_dataset = getdataset(args)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0, # should be 10
    )
    for step, data in enumerate(test_dataloader):

        img_prompt = data['image_data']['image'][0].to(device, dtype=torch.float16) # torch.Size([1, 3, 1, 80, 80])
        aud_prompt = data['audio_data']['audio'].to(device, dtype=torch.float32)

        if args.same_modality:
            img_prompt = data['video_data']['video'][:,:,0:1,:,:].to(device, dtype=torch.float16)

        videos = videogen_pipeline([img_prompt, aud_prompt],
                                   video_length=video_length,
                                   height=image_size,
                                   width=image_size,
                                   num_inference_steps=args.num_sampling_steps,
                                   guidance_scale=args.guidance_scale,
                                   enable_temporal_attentions=not args.force_images,
                                   num_images_per_prompt=1,
                                   mask_feature=True,
                                   ).video
        prompt = data["file_name"][0]
        pred_video =  videos[0].numpy()
        gt_video = data['video_data']['video'][0].detach().cpu().permute((1,2,3,0)).numpy()
        gt_video = cv2.normalize(gt_video, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        pred_gt_video = pred_gt_video = np.concatenate((pred_video, gt_video), axis=2)

        try:
            if args.force_images:
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_image(videos / 255.0, os.path.join(args.save_img_path,
                                                     prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                           nrow=1, normalize=True, value_range=(0, 1))  # t c h w
            else:
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path, "pred", 
                        prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                    ), pred_video,
                    fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
                # gt video
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path, "groundtruth",
                        prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                    ), gt_video,
                    fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
                # pred & gt video
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path, "pred_groundtruth",
                        prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                    ), pred_gt_video,
                    fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    if args.force_images:
        save_image(video_grids / 255.0, os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                   nrow=math.ceil(math.sqrt(len(video_grids))), normalize=True, value_range=(0, 1))
    else:
        video_grids = save_video_grid(video_grids)
        imageio.mimwrite(os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'), video_grids, fps=args.fps, quality=9)

    print('save path {}'.format(args.save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default='65x512x512', choices=['65x512x512', '65x256x256', '17x256x256', "61x80x80", "121x80x80", "241x80x80"])
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--audio_encoder_name", type=str, default='microsoft/speecht5')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--video_data", type=str, required='')
    parser.add_argument("--image_data", type=str, required='')
    parser.add_argument("--audio_data", type=str, required='')
    parser.add_argument("--sample_rate", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--cond_image_num", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=300)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument('--enable_8bit_t5', action='store_true')
    parser.add_argument("--encoder_hidden_states_extend", help="extend the scope of the hidden_states",
                        action="store_true")
    parser.add_argument("--same_modality", help="use the same modality data as condition",
                        action="store_true")
    parser.add_argument("--ae_ckpt_dir", help="the directory of the vae", type=str, default=None)
    args = parser.parse_args()

    main(args)