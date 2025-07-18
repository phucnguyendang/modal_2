import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import modal
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

modal_image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "pillow",
    "fastapi",
    "python-multipart",
    "accelerate",
    "bitsandbytes"
)
app = modal.App("Modal_captioning_inpaint_single_container", image=modal_image)
volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)
CACHE_PATH = "/root/.cache/huggingface/"

@app.cls(
    gpu="A10G",
    image=modal_image,
    volumes={CACHE_PATH: volume},
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
class InstructBlipModel:
    @modal.enter()
    def load_model(self):
        print("Đang khởi tạo container và tải mô hình...")
        self.device = "cuda"

        self.processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl", cache_dir=CACHE_PATH
        )

        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.float16,
            cache_dir=CACHE_PATH,
        ).to(self.device)

        # Biên dịch mô hình để tối ưu tốc độ
        # self.model = torch.compile(model)
        print("Mô hình đã sẵn sàng.")

    def _generate_caption(self, image_base64_list: list[str]) -> list[str]:
        """
        Đây là phương thức nội bộ để tạo caption. Nó không cần decorator
        vì sẽ được gọi trực tiếp từ bên trong class.
        """
        pil_images = [Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB") for b64_data in image_base64_list]
        with torch.inference_mode():
            inputs = self.processor(
                images=pil_images, text="", return_tensors="pt"
            ).to(self.device, torch.float16) 
            
            # Tạo caption
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in generated_texts]

    @modal.asgi_app()
    def fastapi_app(self):
        web_app = FastAPI()

        class ImageCaptioningRequest(BaseModel):
            id: str
            image_data: str

        class ImageCaptioningResponse(BaseModel):
            id: str
            caption: str

        @web_app.post("/captioning", response_model=ImageCaptioningResponse)
        async def caption_single_image(request: ImageCaptioningRequest):
            try:
                captions = self._generate_caption([request.image_data])
                return ImageCaptioningResponse(id=request.id, caption=captions[0])
            except Exception as e:
                print(f"Error processing request: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return web_app
