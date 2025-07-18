import base64
import io
import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F
import torchvision.transforms.v2 as T

modal_image = modal.Image.debian_slim().pip_install(
    "diffusers",
    "torch",
    "pillow",
    "fastapi",
    "python-multipart",
    "torchvision",
    "transformers",
    "accelerate",
    "bitsandbytes"
)
CACHE_PATH = "/root/.cache/stable_diffusion/"
app = modal.App("Modal_inpaint_single_container", image=modal_image)
volume = modal.Volume.from_name("inpaint_volume", create_if_missing=True)

def base64_to_pil(img_base64: str) -> Image.Image:
    """Chuyển đổi chuỗi base64 sang ảnh PIL."""
    return Image.open(io.BytesIO(base64.b64decode(img_base64)))

def pil_to_base64(img: Image.Image) -> str:
    """Chuyển đổi ảnh PIL sang chuỗi base64."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.cls(
    gpu="A10G",
    image=modal_image,
    volumes={CACHE_PATH: volume},
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
class InpaintModel:
    @modal.enter()
    def load_model(self):
        print("Đang khởi tạo container và tải mô hình inpainting...")
        self.device = "cuda"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            cache_dir=CACHE_PATH
        ).to(self.device)
        
        # self.pipe = torch.compile(self.pipe)
        print("Mô hình inpainting đã sẵn sàng.")

    # Các phương thức xử lý mask vẫn là một phần của class
    def expand_mask(self, mask: torch.Tensor, expand : int) -> torch.Tensor:
        if(expand <= 0):
            return mask
        output = mask
        for _ in range(expand):
            output = F.max_pool2d(output, kernel_size=3, stride=1, padding=1)
        return output
    
    def blur_mask(self, mask: torch.Tensor, amount: int) -> torch.Tensor:
        if amount == 0:
            return mask
        if amount % 2 == 0:
            amount += 1
        return T.functional.gaussian_blur(mask, amount)

    def _inpaint_internal(self, image_data, mask_tl_x, mask_tl_y, mask_br_x, mask_br_y, caption):
        import torchvision.transforms as transforms
        
        image = base64_to_pil(image_data).convert("RGB")
        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(mask_tl_x, mask_tl_y), (mask_br_x, mask_br_y)], fill=255)

        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(self.device)

        out_mask = self.expand_mask(mask_tensor, expand=5)
        out_mask = self.blur_mask(out_mask, amount=6)

        with torch.inference_mode():
            result = self.pipe(
                prompt=caption,
                image=image,
                mask_image=transforms.ToPILImage()(out_mask.squeeze(0).cpu()),
                num_inference_steps=20,
                guidance_scale=8.0,
                strength=1.0,
            ).images[0]

        result_tensor = transforms.ToTensor()(result).unsqueeze(0).to(self.device)
        blended = result_tensor * out_mask + image_tensor * (1 - out_mask)
        blended_pil = transforms.ToPILImage()(blended.squeeze(0).cpu())
        
        return pil_to_base64(blended_pil)

    @modal.asgi_app()
    def fastapi_app(self):
        web_app = FastAPI()

        class InpaintRequest_(BaseModel):
            id: str
            image_data: str
            mask_tl_x: int
            mask_tl_y: int
            mask_br_x: int
            mask_br_y: int
            caption: str

        class InpaintResponse_(BaseModel):
            id: str
            image_data: str

        @web_app.post("/inpaint", response_model=InpaintResponse_)
        async def inpaint_endpoint(request: InpaintRequest_):
            try:
                image_base64 = self._inpaint_internal(
                    request.image_data,
                    request.mask_tl_x,
                    request.mask_tl_y,
                    request.mask_br_x,
                    request.mask_br_y,
                    request.caption
                )
                return InpaintResponse_(id=request.id, image_data=image_base64)
            except Exception as e:
                print(f"Lỗi trong quá trình xử lý inpaint: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return web_app
