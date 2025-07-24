# Standard library imports
import os
from datetime import datetime
import subprocess
import time
import uuid
import io
from threading import Thread

# Third-party imports
import numpy as np
import torch
from PIL import Image
import accelerate
import gradio as gr
import spaces
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TextIteratorStreamer,
)

# Local imports
from qwen_vl_utils import process_vision_info

# Set device agnostic code
if torch.cuda.is_available():
    device = "cuda"
elif (torch.backends.mps.is_available()) and (torch.backends.mps.is_built()):
    device = "mps"
else:
    device = "cpu"

print(f"[INFO] Using device: {device}")

# Define supported media extensions
image_extensions = Image.registered_extensions()
video_extensions = (
    "avi",
    "mp4",
    "mov",
    "mkv",
    "flv",
    "wmv",
    "mjpeg",
    "gif",
    "webm",
    "m4v",
    "3gp",
)  # Removed .wav as it's audio, not video


def identify_and_save_blob(blob_path):
    """
    Identifies if the blob is an image or video and saves it with a unique name.
    Returns the saved file path and its media type ("image" or "video").
    """
    try:
        with open(blob_path, "rb") as file:
            blob_content = file.read()

            # Try to identify if it's an image
            try:
                Image.open(
                    io.BytesIO(blob_content)
                ).verify()  # Check if it's a valid image
                extension = ".png"  # Default to PNG for saving
                media_type = "image"
            except (IOError, SyntaxError):
                # If it's not a valid image, assume it's a video
                # We can try to get the actual extension from the blob_path,
                # but for unknown types, MP4 is a good default.
                _, ext = os.path.splitext(blob_path)
                if ext.lower() in video_extensions:
                    extension = ext.lower()
                else:
                    extension = ".mp4"  # Default to MP4 for saving
                media_type = "video"

            # Create a unique filename
            filename = f"temp_{uuid.uuid4()}_media{extension}"
            with open(filename, "wb") as f:
                f.write(blob_content)

            return filename, media_type

    except FileNotFoundError:
        raise ValueError(f"The file {blob_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the file: {e}")


# Model and Processor Loading
# Define models and processors as dictionaries for easy selection
models = {
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    ).eval(),
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    ).eval(),
}

processors = {
    "Qwen/Qwen2.5-VL-7B-Instruct": AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
    ),
    "Qwen/Qwen2.5-VL-3B-Instruct": AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True
    ),
}

DESCRIPTION = "[Qwen2.5-VL Demo](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)"


@spaces.GPU
def run_example(
    video_path: str, text_input: str, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
):
    # if media_input is None:
    #     raise gr.Error("No media provided. Please upload an image or video before submitting.")
    # if model_id is None:
    #     raise gr.Error("No model selected. Please select a model.")

    start_time = time.time()

    # media_path = None
    # media_type = None

    # # Determine if it's an image (numpy array from gr.Image) or a file (from gr.File)
    # if isinstance(media_input, np.ndarray): # This comes from gr.Image
    #     img = Image.fromarray(np.uint8(media_input))
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"image_{timestamp}.png"
    #     img.save(filename)
    #     media_path = os.path.abspath(filename)
    #     media_type = "image"
    # elif isinstance(media_input, str): # This comes from gr.File (filepath)
    #     path = media_input
    #     _, ext = os.path.splitext(path)
    #     ext = ext.lower()

    #     if ext in image_extensions:
    #         media_path = path
    #         media_type = "image"
    #     elif ext in video_extensions:
    #         media_path = path
    #         media_type = "video"
    #     else:
    #         # For blobs or unknown file types, try to identify
    #         try:
    #             media_path, media_type = identify_and_save_blob(path)
    #             print(f"Identified blob as: {media_type}, saved to: {media_path}")
    #         except Exception as e:
    #             print(f"Error identifying blob: {e}")
    #             raise gr.Error("Unsupported media type. Please upload an image (PNG, JPG, etc.) or a video (MP4, AVI, etc.).")
    # else:
    #     raise gr.Error("Unsupported input type for media. Please upload an image or video.")

    # print(f"[INFO] Processing {media_type} from {media_path}")

    model = models[model_id]
    processor = processors[model_id]

    # Construct messages list based on media type
    content_list = []
    # if media_type == "image":
    #     content_list.append({"type": "image", "image": media_path})
    # elif media_type == "video":
    #     content_list.append({"type": "video", "video": media_path, "fps": 8.0}) # Qwen2.5-VL often uses 8fps
    content_list.append({"type": "video", "video": video_path, "fps": 8.0})
    content_list.append({"type": "text", "text": text_input})
    # if text_input:
    #     content_list.append({"type": "text", "text": text_input})
    # else:
    #     # Default prompt if no text_input is provided
    #     content_list.append({"type": "text", "text": "What is in this image/video?"})

    messages = [{"role": "user", "content": content_list}]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(
        messages
    )  # This utility handles both image and video info
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Inference: Generation of the output using streaming
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, **{"skip_special_tokens": True}
    )
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    # Start generation in a separate thread to allow streaming
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer, None  # Yield partial text and None for time until full generation
        # Clean up the temporary file after it's processed (optional, depends on use case)
        # if media_path and os.path.exists(media_path) and "temp_" in os.path.basename(media_path):
        #     os.remove(media_path)

    end_time = time.time()
    total_time = round(end_time - start_time, 2)

    # Final yield with total time
    yield buffer, f"{total_time} seconds"

    # Clean up the temporary file after it's fully processed
    # if media_path and os.path.exists(media_path) and "temp_" in os.path.basename(media_path):
    #     os.remove(media_path)
    #     print(f"[INFO] Cleaned up temporary file: {media_path}")


css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Qwen2.5-VL Input"):
        with gr.Row():
            with gr.Column():
                # Change input to gr.File to accept both image and video
                input_media = gr.Video(label="Input Video")
                text_input = gr.Textbox(
                    label="Text Prompt",
                    value="Describe the camera motion in this video.",
                )
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text", interactive=False)
                time_taken = gr.Textbox(
                    label="Time taken for processing + inference", interactive=False
                )

        submit_btn.click(
            run_example,
            [input_media, text_input],
            [output_text, time_taken],
        )  # Ensure output components match yield order

demo.launch(debug=True)
