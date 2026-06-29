import base64
from pathlib import Path

from app.llm import LLM
from app.schema import Message
from app.tool.base import BaseTool

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageQuery(BaseTool):
    name: str = "image_query"
    description: str = (
        "Answers a question about a local image file by sending it to a vision model "
        "and returning a text description. Use this whenever a task references an image "
        "file (.png, .jpg, .jpeg, .gif, .webp, .bmp). Provide the file path and a "
        "specific question about what to extract or describe from the image."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Absolute or relative path to the local image file.",
            },
            "question": {
                "type": "string",
                "description": "What to extract or describe from the image.",
            },
        },
        "required": ["image_path", "question"],
    }

    async def execute(self, image_path: str, question: str) -> str:
        try:
            path = Path(image_path)
            if not path.exists():
                return f"Error: image file not found: {image_path}"

            mime = _MIME_MAP.get(path.suffix.lower(), "image/jpeg")
            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            data_url = f"data:{mime};base64,{b64}"

            vision_llm = LLM(config_name="vision")
            return await vision_llm.ask_with_images(
                messages=[Message.user_message(question)],
                images=[data_url],
                temperature=0.0,
            )
        except Exception as e:
            return f"Error querying image '{image_path}': {e}"
