import os
import glob
import argparse
from moviepy.editor import VideoFileClip
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import json
import shutil
import tqdm
import scipy
import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio

# valid_sub_id = []
test_sub_id = ["sub001", "sub011", "sub021", "sub031", "sub041", "sub051", "sub061", "sub071"]
# valid_speech_id = []
test_speech_id = ["vcv1", "bvt1", "rainbow_r1", "picture1"]


def get_clip_from_timestamp(audio_timestamps, audio, video, target_length, sampling_rate=16000):
    fullduration = video.duration
    audio_clips, video_clips, time_clips = [], [], []
    start_time_clip = 0
    end_time_clip = 0

    reset_flag = True
    for timestamp in audio_timestamps:
        start_time = timestamp["start"] / sampling_rate
        end_time = timestamp["end"] / sampling_rate
        if not reset_flag:
            end_time_clip = end_time
            segment_length = end_time_clip - start_time_clip
        else:
            start_time_clip = start_time
            end_time_clip = end_time
            segment_length = end_time_clip - start_time_clip
        reset_flag = True if segment_length > target_length else False

        if reset_flag:
            if start_time_clip >= fullduration:
                print(f"Video too short to complete request! Quitting loop..")
                break
            if end_time_clip >= fullduration:
                print(f"Cropping endtime from {end_time_clip} to {fullduration}!")
                end_time_clip = fullduration
            audio_clip = audio.subclip(start_time_clip, end_time_clip)
            video_clip = video.subclip(start_time_clip, end_time_clip)
            audio_clips.append(audio_clip)
            video_clips.append(video_clip)
            time_clips.append([start_time_clip, end_time_clip])
    return audio_clips, video_clips, time_clips


def get_speech_timestamps_and_resample(audio_input, 
                                       audio_vad_model, 
                                       original_sample_rate=16000,
                                       target_sample_rate=44100):

    # number_of_samples = round(len(audio_input) * float(target_sample_rate) / original_sample_rate)
    # downsampled_audio = scipy.signal.resample(audio_input, number_of_samples)
    audio_timestamps = get_speech_timestamps(
        audio_input, audio_vad_model, sampling_rate=original_sample_rate,
    )
    new_audio_timestamps = []
    for audio_timestamp in audio_timestamps:
        new_audio_timestamps.append(
            {"start": int(audio_timestamp["start"] * target_sample_rate / original_sample_rate),
             "end": int(audio_timestamp["end"] * target_sample_rate / original_sample_rate)}
        )
    return new_audio_timestamps
    

def main(args):

    denoiser_model = pretrained.dns64().cuda()
    audio_vad_model = load_silero_vad()
    
    output_audio_dir = os.path.join(args.output_dir, "audio")
    output_video_dir = os.path.join(args.output_dir, "video")
    output_img3d_dir = os.path.join(args.output_dir, "img3d")
    output_img3dsnapshot_dir = os.path.join(args.output_dir, "img3dsnapshot")
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_img3d_dir, exist_ok=True)
    os.makedirs(output_img3dsnapshot_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audio_json"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video_json"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "img3d_json"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "img3dsnapshot_json"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "scripts"), exist_ok=True)

    video_list_txt_train = []
    audio_list_txt_train = []
    img3d_list_txt_train = []
    img3dsnapshot_list_txt_train = []

    video_list_txt_test = []
    audio_list_txt_test = []
    img3d_list_txt_test = []
    img3dsnapshot_list_txt_test = []

    error_list = []

    subfolders = [f.path for f in os.scandir(args.dataset_dir) if f.is_dir()]
    subfolders.sort()
    for subfolder in tqdm.tqdm(subfolders):
        sub_id = subfolder.split("/")[-1]
        video_list = glob.glob(
            os.path.join(
                subfolder,
                "2drt/video/*_video.mp4"
            )
        )

        img3d_path = glob.glob(os.path.join(
            subfolder,
            "3d/recon/*_hold_r*_recon.mat"
        ))[0]
        img3dsnapshot_path = os.path.join(
            subfolder, 
            "3d/snapshot", 
            os.path.basename(img3d_path).replace("recon.mat", "snapshot.png")
        )


        for video_path in video_list:

            video_list_json = []
            audio_list_json = []
            img3d_list_json = []
            img3dsnapshot_list_json = []

            video_name = os.path.basename(video_path)[:-4]
            if sub_id in test_sub_id:
                is_test = True
            else:
                for speech_id in test_speech_id:
                    if speech_id in video_name:
                        is_test = True
                        break
                    else: is_test = False
            try:
                video = VideoFileClip(video_path)
            except:
                continue
            audio = video.audio
            sampling_rate = audio.fps
            # audio.write_audiofile(audio_output_path)
            # video.close()
            # fps = video.fps
            # duration = video.duration
            audio_for_vad_path = video_path.replace("video", "audio").replace(".mp4", ".wav")
            try:
                audio_for_vad = read_audio(audio_for_vad_path, sampling_rate=16000)
            except:
                continue
            wav = convert_audio(audio_for_vad.unsqueeze(0).cuda(), 16000, denoiser_model.sample_rate, denoiser_model.chin)
            with torch.no_grad():
                audio_for_vad = denoiser_model(wav[None])[0]
            # TODO (Xuan)
            audio_timestamps = get_speech_timestamps_and_resample(
                audio_for_vad.cpu(), 
                audio_vad_model,
                target_sample_rate=sampling_rate)

            audio_clips, video_clips, time_clips = get_clip_from_timestamp(
                audio_timestamps, 
                audio, video, 
                target_length=5,
                sampling_rate=sampling_rate,
            )

            if len(audio_clips) == 0 and "postures" not in video_path:
                error_list.append(video_path)

            for a_c, v_c, t_c in zip(audio_clips, video_clips, time_clips):
                audio_name = video_name.replace("video", "audio")
                audio_output_path = os.path.join(
                    output_audio_dir,
                    f"{audio_name}_{t_c[0]:.2f}_{t_c[1]:.2f}.wav"
                )
                try:
                    a_c.write_audiofile(audio_output_path)
                except:
                    continue

                video_output_path = os.path.join(
                    output_video_dir,
                    f"{video_name}_{t_c[0]:.2f}_{t_c[1]:.2f}.mp4"
                )
                try:
                    v_c.write_videofile(video_output_path)
                except:
                    continue

                audio_list_json.append({
                    "path": f"{audio_name}_{t_c[0]:.2f}_{t_c[1]:.2f}.wav", 
                    "cap": "speech"
                })
                video_list_json.append({
                    "path": f"{video_name}_{t_c[0]:.2f}_{t_c[1]:.2f}.mp4", 
                    "cap": "This is a mid-sagittal real-time MRI for vocal tract."
                })
                img3d_list_json.append({
                    "path": os.path.basename(img3d_path), 
                    "cap": "3d_snapshot_mat"
                })
                img3dsnapshot_list_json.append({
                    "path": os.path.basename(img3dsnapshot_path),
                    "cap": "3d_snapshot"
                })
        
            audio_path_json = os.path.join(args.output_dir, "audio_json", f"span_75speakers_{video_name[:-6]}_audio.json")
            with open(audio_path_json, "w") as audio_file_json:
                json.dump(audio_list_json, audio_file_json, indent=4)
            
            video_path_json = os.path.join(args.output_dir, "video_json", f"span_75speakers_{video_name[:-6]}_video.json")
            with open(video_path_json, "w") as video_file_json:
                json.dump(video_list_json, video_file_json, indent=4)
            
            shutil.copy(img3d_path, 
                        os.path.join(output_img3d_dir, os.path.basename(img3d_path)))
            img3d_path_json = os.path.join(args.output_dir, "img3d_json", f"span_75speakers_{video_name[:-6]}_img3d.json")
            with open(img3d_path_json, "w") as img3d_file_json:
                json.dump(img3d_list_json, img3d_file_json, indent=4)

            shutil.copy(img3dsnapshot_path, 
                        os.path.join(output_img3dsnapshot_dir, os.path.basename(img3dsnapshot_path)))
            img3dsnapshot_path_json = os.path.join(args.output_dir, "img3dsnapshot_json", f"span_75speakers_{video_name[:-6]}_img3dsnapshot.json")
            with open(img3dsnapshot_path_json, "w") as img3dsnapshot_file_json:
                json.dump(img3dsnapshot_list_json, img3dsnapshot_file_json, indent=4)

            audio_txt_line = f"{output_audio_dir},{audio_path_json}"
            video_txt_line = f"{output_video_dir},{video_path_json}"
            img3d_txt_line = f"{output_img3d_dir},{img3d_path_json}"
            img3dsnapshot_txt_line = f"{output_img3dsnapshot_dir},{img3dsnapshot_path_json}"

            if is_test:
                audio_list_txt_test.append(audio_txt_line)
                video_list_txt_test.append(video_txt_line)
                img3d_list_txt_test.append(img3d_txt_line)
                img3dsnapshot_list_txt_test.append(img3dsnapshot_txt_line)
            else:
                audio_list_txt_train.append(audio_txt_line)
                video_list_txt_train.append(video_txt_line)
                img3d_list_txt_train.append(img3d_txt_line)
                img3dsnapshot_list_txt_train.append(img3dsnapshot_txt_line)

    def write_txt_file(file_path, lines):
        with open(file_path, 'w') as file:
            for line in lines:
                file.write(line + '\n')

    iter_index = "all"
    write_txt_file(os.path.join(args.output_dir, "scripts", f"audio_mri_train_{iter_index}.txt"), audio_list_txt_train)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"video_mri_train_{iter_index}.txt"), video_list_txt_train)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"img3d_mri_train_{iter_index}.txt"), img3d_list_txt_train)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"img3dsnapshot_mri_train_{iter_index}.txt"), img3dsnapshot_list_txt_train)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"audio_mri_test_{iter_index}.txt"), audio_list_txt_test)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"video_mri_test_{iter_index}.txt"), video_list_txt_test)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"img3d_mri_test_{iter_index}.txt"), img3d_list_txt_test)
    write_txt_file(os.path.join(args.output_dir, "scripts", f"img3dsnapshot_mri_test_{iter_index}.txt"), img3dsnapshot_list_txt_test)
    write_txt_file(os.path.join(args.output_dir, f"error_list_{iter_index}.txt"), error_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='directory of dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='directory of output')

    args = parser.parse_args()
    
    main(args)