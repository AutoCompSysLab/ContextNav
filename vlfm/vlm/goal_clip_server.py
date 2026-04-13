import torch
from transformers import AutoProcessor, AutoModel
from .server_wrapper import ServerMixin, host_model, str_to_image
from ..utils.func import longclip_pos_embeddings  # from GOAL repository

class GoalClip:
    """GOAL-CLIP image-text similarity model."""
    def __init__(self, ckpt_path: str, model_name: str = "openai/clip-vit-large-patch14", max_token: int = 128, device="cuda"):
        # Load base CLIP
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Extend positional embeddings for longer text
        longclip_pos_embeddings(self.model, max_token)
        # Load GOAL fine-tuned weights
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def cosine(self, image_np, text: str) -> float:
        img_tensor = self.processor(images=image_np, return_tensors="pt").pixel_values.to(self.device)
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        token = text_inputs["input_ids"].to(self.device)

        with torch.inference_mode():
            out = self.model(pixel_values=img_tensor, input_ids=token)
            img_feat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt_feat = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).item()
        return sim

class GoalClipServer(ServerMixin, GoalClip):
    """Flask server request handler wrapper."""

    def __init__(self, ckpt_path: str, model_name="openai/clip-vit-large-patch14", max_token=248):
        super().__init__(ckpt_path=ckpt_path, model_name=model_name, max_token=max_token)

    def process_payload(self, payload: dict):
        image = str_to_image(payload["image"])
        text = payload["text"]
        sim = self.cosine(image, text)
        return {"response": sim}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    model = GoalClipServer(ckpt_path=args.ckpt)
    host_model(model, name="goal_clip", port=args.port)
