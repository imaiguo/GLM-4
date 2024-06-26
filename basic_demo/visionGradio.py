import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import gradio as gr
from threading import Thread
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer, AutoModel, BitsAndBytesConfig
)
from PIL import Image
from loguru import logger

MODEL_PATH = "/opt/Data/ModelWeight/THUDM/glm-4v-9b"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    encode_special_tokens=True
)

# int4量化
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto", # 多卡部署
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
).eval()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def get_image(image_path=None):
    if image_path:
        logger.debug(f"image_path:[{image_path}]")
        return Image.open(image_path).convert("RGB")
    return None

def chatbot(image_path=None, assistant_prompt=""):
    logger.debug(f"input->:{assistant_prompt}")
    image = get_image(image_path)

    messages = [
        {"role": "assistant", "content": assistant_prompt},
        {"role": "user", "content": "", "image": image}
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(next(model.parameters()).device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generate_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.6,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),
        "repetition_penalty": 1.2,
        "eos_token_id": [151329, 151336, 151338],
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    response = ""
    for new_token in streamer:
        if new_token:
            response += new_token

    logger.debug(f"response->:{response}")
    return image, response.strip()

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    demo.title = "4V-9B 图像识别演示"
    with gr.Row():
        with gr.Column():
            image_path_input = gr.File(label="图片上传", type="filepath")
            image_output = gr.Image(label="图片预览")

        with gr.Column():
            chatbot_output = gr.Textbox(label="4V-9B模型回答:", lines=20)
            with gr.Row():
                with gr.Column(scale=9):
                    assistant_prompt_input = gr.Textbox(show_label=False, placeholder="请输入您的问题,刷新页面可清除历史", lines=1, container=False)

                with gr.Column(min_width=1, scale=1):
                    submit_button = gr.Button("提交", min_width=1, scale=1, variant="primary")

    submit_button.click(chatbot, inputs=[image_path_input, assistant_prompt_input], outputs=[image_output, chatbot_output])
    assistant_prompt_input.submit(chatbot, inputs=[image_path_input, assistant_prompt_input], outputs=[image_output, chatbot_output])

demo.launch(server_name="0.0.0.0", server_port=8000, inbrowser=False, share=False)