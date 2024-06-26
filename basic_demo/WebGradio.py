"""
This script creates an interactive web demo for the GLM-4-9B model using Gradio,
a Python library for building quick and easy UI components for machine learning models.
It's designed to showcase the capabilities of the GLM-4-9B model in a user-friendly interface,
allowing users to interact with the model through a chat-like interface.
"""

import os

from loguru import logger
from pathlib import Path
from threading import Thread
from typing import Union
from dotenv import load_dotenv

load_dotenv()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import gradio as gr
import torch
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

BindLocalIP = os.getenv('LocalIP')
BindPort = os.getenv('BindPort')
GradioUser = os.getenv('GradioUser')
GradioPassword = os.getenv('GradioPassword')

print(f"BindLocalIP: {BindLocalIP}")
print(f"BindPort: {BindPort}")
print(f"GradioUser: {GradioUser}")
print(f"GradioPassword: {GradioPassword}")

Device = 'auto'
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

MODEL_PATH = '/opt/Data/ModelWeight/THUDM/glm-4-9b-chat'
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

PROMPT = "You are Intelligent Customer Service Blue, carefully analyzing the user's input and providing detailed and accurate answers.你是智能客服小蓝，仔细分析用户的输入，并作详细又准确的回答，记住使用中文回答问题。"

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map=Device
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map=Device
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
    )
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(history):
    stop = StopOnTokens()
    messages = []
    if PROMPT:
        messages.append({"role": "system", "content": PROMPT})
    for idx, (user_msg, model_msg) in enumerate(history):
        # if PROMPT and idx == 0:
        #     continue
        if idx == len(history) - 1 and not model_msg:
            logger.debug(f"input->:{user_msg}")
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            logger.debug(f"input->:{user_msg}")
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": 8192,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.6,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    print("answer->:", end='', flush=True)
    for new_token in streamer:
        if new_token:
            history[-1][1] += new_token
            print(new_token.strip(), end='', flush=True)
        yield history

with gr.Blocks(title = "智能客服小蓝", css="footer {visibility: hidden}")  as demo:
    # gr.HTML("""<h1 align="center">GLM-4-9B 聊天演示</h1>""")
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=800,
        show_copy_button = False,
        layout= "bubble",
        avatar_images=("./image/Einstein.jpg", "./image/openai.png"))

    with gr.Row():
        with gr.Column(scale=9):
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题,刷新页面可清除历史", lines=1, container=False)

        with gr.Column(min_width=1, scale=1):
            submitBtn = gr.Button("提交", variant="primary")

    def user(query, history):
        return "", history + [[parse_text(query), ""]]

    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(predict, [chatbot], chatbot)
    user_input.submit(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(predict, [chatbot], chatbot)

demo.queue()
demo.launch(server_name=BindLocalIP, server_port=int(BindPort), inbrowser=False, share=False)
