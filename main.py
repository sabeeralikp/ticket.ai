from typing import Optional
import torch
from PIL import Image
import io
import base64
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from huggingface_hub import snapshot_download
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request, Form, Response
import httpx
from contextlib import asynccontextmanager

import os
import asyncio

os.environ.update({'OLLAMA_HOST': '0.0.0.0', 'GIN_MODE': 'release'})

async def run(cmd):
  '''
  run is a helper function to run subcommands asynchronously.
  '''
  print('>>> starting', *cmd)
  p = await asyncio.subprocess.create_subprocess_exec(
      *cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
  )

  async def pipe(lines):
    async for line in lines:
      print(line.strip().decode('utf-8'))

  await asyncio.gather(
      pipe(p.stdout),
      pipe(p.stderr),
  )

repo_id = "microsoft/OmniParser-v2.0"  # HF repo
local_dir = "weights"  # Target local directory

# Download the entire repository
snapshot_download(repo_id=repo_id, local_dir=local_dir)

print(f"Repository downloaded to: {local_dir}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Ollama server when app starts
    await run(['ollama', 'serve'])
    yield

app = FastAPI(lifespan=lifespan)

# ======== Ollama Proxy Endpoint ======== #
@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT"])
async def ollama_proxy(path: str, request: Request):
    async with httpx.AsyncClient(base_url="http://localhost:11434") as client:
        response = await client.request(
            method=request.method,
            url=path,
            headers=dict(request.headers),
            content=await request.body()
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )

DEVICE = torch.device('cuda')

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption")



@torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold=0.05,
    iou_threshold=0.1,
    use_paddleocr=False,
    imgsz=640
) -> Optional[Image.Image]:

    # image_save_path = 'imgs/saved_image_demo.png'
    # image_input.save(image_save_path)
    # image = Image.open(image_save_path)
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    # parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    # parsed_content_list = str(parsed_content_list)
    return image, parsed_content_list

def save_image(out_image: Image):
    try:
        out_image.save(f"images/{datetime.now()}.png")
    except Exception as e:
        print(e)

@app.post("/detect/")
async def detect(
    background_task:BackgroundTasks,
    image: UploadFile = File(...)):
    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    out_image, parsed_content = process(pil_image)
    background_task.add_task(save_image, out_image)
    return parsed_content