import torch
import numpy as np
import cv2
import base64
from PIL import Image
from captum.attr import LayerGradCam
from model import model

# -----------------------------
# Grad-CAM Overlay
# -----------------------------
def generate_gradcam_overlay(image_pil, image_tensor, target_class=None):
    """
    Generates a Grad-CAM overlay on the original image
    """
    target_layer = model.features[-1]  # last conv layer
    gradcam = LayerGradCam(model, target_layer)

    if target_class is None:
        with torch.no_grad():
            outputs = model(image_tensor)
            target_class = torch.argmax(outputs, dim=1).item()

    attribution = gradcam.attribute(image_tensor, target=target_class)
    heatmap = attribution.squeeze().detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, image_pil.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original image to OpenCV format (BGR)
    original = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Overlay heatmap on original
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")

# -----------------------------
# Intrinsic Activation Maps
# -----------------------------
def generate_intrinsic_maps(image_tensor, layers=None, max_maps=3):
    """
    Extracts intermediate feature maps from DenseNet for hybrid XAI
    Returns a list of base64-encoded images
    """
    if layers is None:
        # select last 3 dense blocks for intrinsic maps
        layers = [model.features.denseblock1,
                  model.features.denseblock2,
                  model.features.denseblock3]

    intrinsic_maps_base64 = []

    x = image_tensor
    for layer in layers:
        x = layer(x)
        # take the first 'max_maps' channels
        fmap = x[0, :max_maps, :, :].detach().cpu().numpy()
        # normalize and convert each channel to image
        for i in range(fmap.shape[0]):
            heatmap = fmap[i]
            heatmap -= heatmap.min()
            heatmap /= heatmap.max() + 1e-8
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            _, buffer = cv2.imencode(".png", heatmap)
            intrinsic_maps_base64.append(base64.b64encode(buffer).decode("utf-8"))

    return intrinsic_maps_base64
