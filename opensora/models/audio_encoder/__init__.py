import torch
from torch import nn
from transformers import T5EncoderModel, CLIPModel, CLIPProcessor
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from transformers import ClapModel, ClapProcessor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import torch.nn.functional as F
from .speechlm.SpeechLM import SpeechLMConfig, SpeechLM

from opensora.utils.utils import get_precision


class SpeechLMWrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(SpeechLMWrapper, self).__init__()
        self.checkpoint = torch.load('models_zoo/speechlmp_base_checkpoint_clean.pt')
        cfg = SpeechLMConfig(self.checkpoint['cfg']['model'])
        self.audio_encoder = SpeechLM(cfg)
        self.audio_encoder.load_state_dict(self.checkpoint['model'])
        self.audio_encoder.eval()

    def forward(self, wav_input_16khz):
        normalize = self.checkpoint['cfg']['task']['normalize']  # False for base model, True for large model
        if normalize:
            wav_input_16khz = F.layer_norm(wav_input_16khz[0], wav_input_16khz[0].shape).unsqueeze(0)

        # extract the representation of last layer
        audio_encoder_embs = self.audio_encoder.extract_features(wav_input_16khz)[0]

        batch_size, embs_len = audio_encoder_embs.shape[0], audio_encoder_embs.shape[1]
        cond_mask = torch.ones(batch_size, embs_len)
        return audio_encoder_embs.detach(), cond_mask


class SpeechT5Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(SpeechT5Wrapper, self).__init__()
        self.audio_encoder = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr").get_encoder().eval()
        
    def forward(self, input):
        audio_encoder_embs = self.audio_encoder(input)[0] # B * 63 * 768
        batch_size = audio_encoder_embs.shape[0]
        embs_len = audio_encoder_embs.shape[1]
        cond_mask = torch.ones(batch_size, embs_len)
        return audio_encoder_embs.detach(), cond_mask


class CLAPWrapper(nn.Module):
    def __init__(self, args):
        super(CLAPWrapper, self).__init__()
        self.model_name = args.audio_encoder_name
        dtype = get_precision(args)
        model_kwargs = {'cache_dir': args.cache_dir, 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
        self.text_enc = CLIPModel.from_pretrained(self.model_name, **model_kwargs).eval()

    def forward(self, input_ids, attention_mask): 
        text_encoder_embs = self.text_enc.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return text_encoder_embs.detach()


audio_encoder = {
    'microsoft/speechlm': SpeechLMWrapper,
    'microsoft/speecht5': SpeechT5Wrapper,
    'clap': CLAPWrapper
}

def get_audio_warpper(audio_encoder_name):
    """deprecation"""
    audio_enc = audio_encoder.get(audio_encoder_name, None)
    assert audio_enc is not None
    return audio_enc
