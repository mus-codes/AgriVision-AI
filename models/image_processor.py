from PIL import Image, ImageOps
from torchvision import transforms


IMAGE_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def validate_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image.")

    # Validate format only when the image comes from a file.
    if image.format and image.format.upper() not in ["JPEG", "PNG", "WEBP"]:
        raise ValueError("Unsupported image format.")

    return True


def preprocess_image(image):
    validate_image(image)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image_tensor = image_transform(image)

    return image_tensor.unsqueeze(0)