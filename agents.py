import base64
import json
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


class VisionAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def run(self, base64_image):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Classify the main physical object in the center of this cropped image. "
                                "This crop comes from a robot RGB-D camera. "
                                "Return ONLY valid JSON."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_object_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short object name, for example green cube, red-gray pliers, black cable.",
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "cube",
                                    "sphere",
                                    "cylinder",
                                    "tool",
                                    "cable",
                                    "box",
                                    "bottle",
                                    "part",
                                    "robot_arm",
                                    "unknown",
                                ],
                            },
                            "colors": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "red",
                                        "green",
                                        "blue",
                                        "yellow",
                                        "white",
                                        "black",
                                        "gray",
                                        "orange",
                                        "purple",
                                        "brown",
                                        "unknown",
                                    ],
                                },
                            },
                            "graspable": {
                                "type": "boolean",
                                "description": "True if a small robot gripper can probably pick it up.",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "name",
                            "category",
                            "colors",
                            "graspable",
                            "confidence",
                        ],
                    },
                },
            },
            max_tokens=120,
            temperature=0.0,
        )

        content = response.choices[0].message.content

        # Validate that it is JSON before returning it.
        parsed = json.loads(content)

        return json.dumps(parsed)


vision_prompt = """
You are a robotic vision classifier for a ROS 2 robotic arm digital twin.

You receive a cropped RGB image of ONE object candidate found by depth segmentation.
Classify the main physical object in the center of the crop.

Important rules:
- Focus on the object in the center.
- Ignore background, table, hand, robot arm parts, wires in the background, and shadows unless they are the main centered object.
- If the crop is part of the robot arm or motor/display, category must be "robot_arm" and graspable must be false.
- If the object has multiple colors, include multiple colors, for example ["red", "gray"].
- Use "unknown" when unsure.
- Keep name short, for example:
  - "green cube"
  - "red-gray pliers"
  - "black cable"
  - "gray tool"
  - "unknown object"

Return only the structured JSON object.
"""

perception_agent = VisionAgent(vision_prompt)