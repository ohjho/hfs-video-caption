import spaces
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- Installing Flash Attention for ZeroGPU is special --- #
import subprocess

subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)
# --- now we got Flash Attention ---#

# The model is trained on 8.0 FPS which we recommend for optimal inference


@spaces.GPU(duration=30)
def load_model(
    model_name: str = "chancharikm/qwen2.5-vl-7b-cam-motion-preview",
    use_flash_attention: bool = True,
):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        if use_flash_attention
        else Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda",
        )
    )
    return model


@spaces.GPU(duration=120)
def inference(video_path: str):
    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 8.0,
                },
                {"type": "text", "text": "Describe the camera motion in this video."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Video(label="Input Video"),
    ],
    outputs=gr.JSON(label="Output JSON"),
    title="",
    api_name="video_inference",
)
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
