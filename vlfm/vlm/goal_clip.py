from .server_wrapper import send_request

class GoalClipClient:
    """Client that sends requests to the GOAL-CLIP server."""
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/goal_clip"

    def cosine(self, image, text: str) -> float:
        resp = send_request(self.url, image=image, text=text)
        return float(resp["response"])
