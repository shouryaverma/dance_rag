# TODO: compute similarity score based on text features
#     Args:
# 1. load the dataset
# 2. load music features
# 3. load text features
# 4. compute similarity score
import os
from functools import partial
from pathlib import Path

import jukemirlib
import numpy as np
from tqdm import tqdm
import clip
import torch
from datasets.text2duet import Text2Duet
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

FPS = 10
LAYER = 66

global_device = "cuda"
# jukemirlib.setup_models(cache_dir="/depot/bera89/data/li5280/project/juke_cache", device="cuda:0")
clip_model, _ = clip.load("ViT-L/14@336px", device=global_device, jit=False)

def extract_music_feature(fpath):
    # fpath: music file path
    with torch.no_grad():
        audio = jukemirlib.load_audio(fpath)
        reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)
    # TODO: check feature size
    print('music feature shape')
    print(reps[LAYER].shape)
    # T, 4800
    return reps[LAYER]

def extract_text_feature(ori_text):
    # ori_text: text    
    if len(ori_text) > 77:
        ori_text = ori_text[:77]
    text = clip.tokenize(ori_text).to(global_device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    # global text features
    return text_features.cpu().numpy()

def process():
    music_root = '/scratch/gilbreth/gupta596/MotionGen/Text2Duet/data_split/music'
    motion_root = '/scratch/gilbreth/gupta596/MotionGen/Text2Duet/data_split/motion'
    text_root = '/scratch/gilbreth/gupta596/MotionGen/Text2Duet/data_split/text'
    print('loading dataset')
    t2d = Text2Duet(music_root, motion_root, text_root, split='all', dtype='pos3d', music_dance_rate=1)
    # text feature dict
    # music feature dict
    music_feature_dict = {}
    text_feature_dict = {}
    print(f'dataset loaded, length = {len(t2d)}')
    # first round: extract features
    list_of_takes = []
    take_text = {}
    take_length = {}
    take_motion_concatenated = {}
    take_clip_feature = {}
    for i in range(len(t2d)):
        item = t2d[i]
        take = item['fname']
        music_wav_path = item['music_wav_path']
        text = item['text']
        # music_feature = extract_music_feature(music_wav_path)
        text_feature = extract_text_feature(text)
        music_feature_dict[take] = ""
        text_feature_dict[take] = text_feature
        list_of_takes.append(take)

        take_text[take] = text
        take_clip_feature[take] = text_feature
        take_length[take] = item['length']
        take_motion_concatenated[take] = np.concatenate([item['motion1'], item['motion2']], axis=-1) # T, F

    print('first round done')
    similarity_dict = {}  # take-> [takes]
    for take in tqdm(list_of_takes):
        text_feature = text_feature_dict[take]
        music_feature = music_feature_dict[take]
        similarity_list = []

        for other_take, other_music_feature in music_feature_dict.items():
            if take == other_take:
                continue  # Skip self-comparison

            other_text_feature = text_feature_dict[other_take]

            # Compute cosine similarity
            # music_similarity = cosine_similarity(music_feature.reshape(1, -1), other_music_feature.reshape(1, -1))[0, 0]
            text_similarity = cosine_similarity(text_feature.reshape(1, -1), other_text_feature.reshape(1, -1))[0, 0]

            # total_similarity = music_similarity + text_similarity
            total_similarity = text_similarity
            similarity_list.append(other_take)

        # Sort by similarity in descending order and keep the top 5
        similarity_list.sort(reverse=True, key=lambda x: x[0])
        similarity_dict[take] = similarity_list[:5]  # Save top 5 similarities with corresponding takes
    
    overall_dict = {
        'captions': take_text,
        'm_lengths': take_length,
        'motions': take_motion_concatenated,
        'clip_seq_features': take_clip_feature,
        'indexes': similarity_dict}
    print('saving the dict')
    # sample_takes = list(similarity_dict.keys())[:5]
    # for take in sample_takes:
    #     print(f"Take: {take}")
    #     print(similarity_dict[take])

    np.savez('RAG_dict.npz', overall_dict=overall_dict)

if __name__ == '__main__':
    process()

# @SUBMODULES.register_module()
# class ReMoDiffuseTransformer(DiffusionTransformer):
#     def __init__(self,
#                  retrieval_cfg=None,
#                  scale_func_cfg=None,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.database = RetrievalDatabase(**retrieval_cfg)
#         self.scale_func_cfg = scale_func_cfg
        
#     def scale_func(self, timestep):
#         coarse_scale = self.scale_func_cfg['coarse_scale']
#         w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
#         if timestep > 100:
#             if random.randint(0, 1) == 0:
#                 output = {
#                     'both_coef': w,
#                     'text_coef': 0,
#                     'retr_coef': 1 - w,
#                     'none_coef': 0
#                 }
#             else:
#                 output = {
#                     'both_coef': 0,
#                     'text_coef': w,
#                     'retr_coef': 0,
#                     'none_coef': 1 - w
#                 }
#         else:
#             both_coef = self.scale_func_cfg['both_coef']
#             text_coef = self.scale_func_cfg['text_coef']
#             retr_coef = self.scale_func_cfg['retr_coef']
#             none_coef = 1 - both_coef - text_coef - retr_coef
#             output = {
#                 'both_coef': both_coef,
#                 'text_coef': text_coef,
#                 'retr_coef': retr_coef,
#                 'none_coef': none_coef
#             }
#         return output
            
#     def get_precompute_condition(self, 
#                                  text=None,
#                                  motion_length=None,
#                                  xf_out=None,
#                                  re_dict=None,
#                                  device=None,
#                                  sample_idx=None,
#                                  clip_feat=None,
#                                  **kwargs):
#         if xf_out is None:
#             xf_out = self.encode_text(text, clip_feat, device)
#         output = {'xf_out': xf_out}
#         if re_dict is None:
#             re_dict = self.database(text, motion_length, self.clip, device, idx=sample_idx)
#         output['re_dict'] = re_dict
#         return output

#     def post_process(self, motion):
#         return motion

#     def forward_train(self, h=None, src_mask=None, emb=None, xf_out=None, re_dict=None, **kwargs):
#         B, T = h.shape[0], h.shape[1]
#         cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
#         for module in self.temporal_decoder_blocks:
#             h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=cond_type, re_dict=re_dict)

#         output = self.out(h).view(B, T, -1).contiguous()
#         return output
    
#     def forward_test(self, h=None, src_mask=None, emb=None, xf_out=None, re_dict=None, timesteps=None, **kwargs):
#         B, T = h.shape[0], h.shape[1]
#         both_cond_type = torch.zeros(B, 1, 1).to(h.device) + 99
#         text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
#         retr_cond_type = torch.zeros(B, 1, 1).to(h.device) + 10
#         none_cond_type = torch.zeros(B, 1, 1).to(h.device)
        
#         all_cond_type = torch.cat((
#             both_cond_type, text_cond_type, retr_cond_type, none_cond_type
#         ), dim=0)
#         h = h.repeat(4, 1, 1)
#         xf_out = xf_out.repeat(4, 1, 1)
#         emb = emb.repeat(4, 1)
#         src_mask = src_mask.repeat(4, 1, 1)
#         if re_dict['re_motion'].shape[0] != h.shape[0]:
#             re_dict['re_motion'] = re_dict['re_motion'].repeat(4, 1, 1, 1)
#             re_dict['re_text'] = re_dict['re_text'].repeat(4, 1, 1, 1)
#             re_dict['re_mask'] = re_dict['re_mask'].repeat(4, 1, 1)
#         for module in self.temporal_decoder_blocks:
#             h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask, cond_type=all_cond_type, re_dict=re_dict)
#         out = self.out(h).view(4 * B, T, -1).contiguous()
#         out_both = out[:B].contiguous()
#         out_text = out[B: 2 * B].contiguous()
#         out_retr = out[2 * B: 3 * B].contiguous()
#         out_none = out[3 * B:].contiguous()
        
#         coef_cfg = self.scale_func(int(timesteps[0]))
#         both_coef = coef_cfg['both_coef']
#         text_coef = coef_cfg['text_coef']
#         retr_coef = coef_cfg['retr_coef']
#         none_coef = coef_cfg['none_coef']
#         output = out_both * both_coef + out_text * text_coef + out_retr * retr_coef + out_none * none_coef
#         return output
