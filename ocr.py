from ollama import chat
from pathlib import Path
import base64
import time

# Pass in the path to the image
# path = input('Please enter the path to the image: ')
path = r"C:\Users\Liviu\Desktop\liviumit\Assist\Contests\fs_test\agents.png"
# You can also pass in base64 encoded image data
img = base64.b64encode(Path(path).read_bytes()).decode()
# or the raw bytes
# img = Path(path).read_bytes()
start_time = time.time()
response = chat(
    model='glm-ocr',
    # model="glm-ocr:bf16",
    messages=[
        {
            "role": "system",
            "content": "You explain the image in a few sentences",
        },
        {
            "role": "user",
            "content": "Explain the image in a few sentences",
            "images": [img],
        }
    ],
    stream=True,
)
for chunk in response:
    print(chunk.message.content, end="", flush=True)
print()
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
