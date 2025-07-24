from statistics import quantiles
import spaces, ffmpeg, os, sys, torch
import gradio as gr
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)

# --- Installing Flash Attention for ZeroGPU is special --- #
import subprocess

subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)
# --- now we got Flash Attention ---#

# Set target DEVICE and DTYPE
# For maximum memory efficiency, use bfloat16 if your GPU supports it, otherwise float16.
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use "auto" to let accelerate handle device placement (GPU, CPU, disk)
DEVICE = "auto"
logger.info(f"Device: {DEVICE}, dtype: {DTYPE}")


def get_fps_ffmpeg(video_path: str):
    probe = ffmpeg.probe(video_path)
    # Find the first video stream
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise ValueError("No video stream found")
    # Frame rate is given as a string fraction, e.g., '30000/1001'
    r_frame_rate = video_stream["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    return num / denom


def load_model(
    model_name: str = "chancharikm/qwen2.5-vl-7b-cam-motion-preview",
    use_flash_attention: bool = True,
    apply_quantization: bool = True,
):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # especially in multi-image and video scenarios.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load model weights in 4-bit
        bnb_4bit_quant_type="nf4",  # Use NF4 quantization (or "fp4")
        bnb_4bit_compute_dtype=DTYPE,  # Perform computations in bfloat16/float16
        bnb_4bit_use_double_quant=True,  # Optional: further quantization for slightly more memory saving
    )

    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            attn_implementation="flash_attention_2",
            device_map=DEVICE,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if apply_quantization else None,
        )
        if use_flash_attention
        else Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if apply_quantization else None,
        )
    )
    # Set model to evaluation mode for inference (disables dropout, etc.)
    model.eval()
    return model


def load_processor(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    return AutoProcessor.from_pretrained(
        model_name,
        device_map=DEVICE,
        use_fast=True,
        torch_dtype=DTYPE,
    )


@spaces.GPU(duration=120)
def inference(
    video_path: str,
    prompt: str = "Describe the camera motion in this video.",
    use_flash_attention: bool = True,
    apply_quantization: bool = True,
):
    # default processor
    processor = load_processor()
    model = load_model(
        use_flash_attention=use_flash_attention, apply_quantization=apply_quantization
    )

    # The model is trained on 8.0 FPS which we recommend for optimal inference
    fps = get_fps_ffmpeg(video_path)
    logger.info(f"{os.path.basename(video_path)} FPS: {fps}")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    # This prevents PyTorch from building the computation graph for gradients,
    # saving a significant amount of memory for intermediate activations.
    with torch.no_grad():
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # fps=fps,
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
        gr.Textbox(label="Prompt", value="Describe the camera motion in this video."),
        gr.Checkbox(label="Use Flash Attention", value=False),
        gr.Checkbox(label="Apply Quantization", value=True),
    ],
    outputs=gr.JSON(label="Output JSON"),
    title="",
    api_name="video_inference",
)
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
