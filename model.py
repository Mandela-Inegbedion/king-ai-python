import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# -------------------------
# Device configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load DenseNet121 model
# -------------------------
def load_model():
    """
    Loads a DenseNet121 model adapted for binary pneumonia classification
    """
    model = models.densenet121(pretrained=True)

    # Replace classifier for binary classification (Pneumonia / Normal)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)

    model.eval()
    model.to(device)
    return model

model = load_model()

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# Prediction function
# -------------------------
def predict(image: Image.Image):
    """
    Runs inference on a chest X-ray image
    """
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    label_map = {
        0: "Normal",
        1: "Pneumonia"
    }

    return {
        "label": label_map[predicted_class.item()],
        "confidence": confidence.item(),
        "image_tensor": image_tensor
    }
