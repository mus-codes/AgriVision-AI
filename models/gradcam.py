import torch
import torch.nn.functional as F
import numpy as np

from models.image_processor import preprocess_image


def generate_gradcam(image, model, class_mapping, class_name=None):

    device = next(model.parameters()).device

    # Preprocess image
    image_tensor = preprocess_image(image).to(device)

    # Target the final EfficientNet feature layer
    target_layer = model.features[-1]

    activations = {}

    # Capture feature maps
    def forward_hook(module, input, output):
        activations["features"] = output

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()

        # Grad-CAM requires gradient computation
        with torch.enable_grad():

            image_tensor.requires_grad_(True)

            # Forward pass
            outputs = model(image_tensor)

            # Determine target class
            if class_name is None:
                class_index = outputs.argmax(dim=1).item()
            else:
                class_index = class_mapping[class_name]

            # Get score for selected class
            score = outputs[0, class_index]

            # Calculate gradients with respect to feature maps
            gradients = torch.autograd.grad(
                outputs=score,
                inputs=activations["features"],
                retain_graph=False
            )[0]

            feature_maps = activations["features"].detach()

            gradients = gradients.detach()

        # Calculate channel-wise importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Generate Grad-CAM heatmap
        cam = torch.sum(
            weights * feature_maps,
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        # Resize heatmap to original model input dimensions
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # Normalize heatmap
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cam.cpu().numpy()

        # Return heatmap and predicted class
        reverse_mapping = {
            index: name
            for name, index in class_mapping.items()
        }

        predicted_class = reverse_mapping[class_index]

        return {
            "heatmap": heatmap,
            "class_name": predicted_class,
            "class_index": class_index
        }

    finally:
        hook.remove()