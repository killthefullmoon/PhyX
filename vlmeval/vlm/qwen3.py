from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from .base import BaseModel
from ..smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen3(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        max_new_tokens=8000,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        **kwargs: dict[str, str | int | float | bool | None],
    ):
        super().__init__()
        self.enable_thinking = kwargs.get('enable_thinking', False)
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens if not self.enable_thinking else 32768,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None


        from transformers import AutoModelForCausalLM, AutoTokenizer
        MODEL_CLS = AutoModelForCausalLM
        self.processor = AutoTokenizer.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        if auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
            )
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
            )
            self.model.cuda().eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f'Unsupported type: {s["type"]}')
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):

        messages = []
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
        for idx in range(len(messages)):
            assert len(messages[idx]['content']) == 1, f'Only support one content per message, but got {len(messages[idx]["content"])}: {messages[idx]}'
            messages[idx]['content'] = messages[idx]['content'][0]['text']
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        inputs = self.processor([text], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist() 
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.processor.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        out = self.processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        response = out
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response