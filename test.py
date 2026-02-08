import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tkinter import Tk, filedialog

# 1. Define the classes (CIFAR-10 default order)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image_path, model_path="best.pt"):
    # 2. Recreate the exact same model architecture
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 10)

    # 3. Load your saved weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model.eval() # Set to evaluation mode

    # 4. Define the same transforms used in validation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Load and transform the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0) # Add batch dimension (1, 3, 224, 224)

    # 6. Predict!
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, index = torch.max(probabilities, dim=1)

    print(f"Prediction: {classes[index.item()]} ({confidence.item()*100:.2f}%)")

def choose_image():
    """Open a file dialog to select an image"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

if __name__ == "__main__":
    image_path = choose_image()
    
    if image_path:
        print(f"Selected image: {image_path}")
        predict(image_path)
    else:
        print("No image selected!")