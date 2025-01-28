import vllm 
import os
from urllib.request import urlretrieve
import base64
from PIL import ImageOps, Image
import io
import openai
import time

os.environ["VLLM_TORCH_PROFILER_DIR"] = "/root/dev/traces"


llm = vllm.LLM(model="llava-hf/llava-1.5-7b-hf")

# Refer to the HuggingFace repo for the correct format to use
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

# Load the image using PIL.Image
# image_size = (1920, 1080) # 148819 chars
image_size = (512, 512)
# IMAGE_SIZE_LIMIT = (1, 1) # 947 chars

def encode_image(image):
    buf = io.BytesIO()
    ImageOps.contain(image.convert("RGB"), image_size).save(
        buf, format="JPEG"
    )
    return image
    # return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

urlretrieve("https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg", "Mona_Lisa.jpg")
# images = [Image.open("Mona_Lisa.jpg")] * 100
images = [encode_image(Image.open("Mona_Lisa.jpg"))] * 1000

# llm.start_profile()

start_time = time.perf_counter()

messages = [{"prompt": prompt,
             "multi_modal_data": {"image": image},
             } for image in images]

outputs = llm.generate(messages, 
                    #    sampling_params=vllm.SamplingParams(max_tokens=1)
                    )

duration = time.perf_counter() - start_time

# llm.stop_profile()



for o in outputs:
    generated_text = o.outputs[0].text
    print(f"\nOutput: {generated_text}\n")

print(f"\nduration: {duration}\n")