from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def image_to_caption(image_path):
    # Model and processor load
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Image load
    image = Image.open(image_path).convert("RGB")

    # Image processing
    inputs = processor(image, return_tensors="pt")

    # Create caption
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

if __name__ == "__main__":
    print("Enter image path:")
    image_path = input("> ").strip()

    try:
        caption = image_to_caption(image_path)
        print("\n=== caption ===")
        print(caption)
    except Exception as e:
        print(f"error: {e}")
