import torch
import cv2
import numpy as np
from torchvision import transforms, datasets
from PIL import Image

from model import get_model
from gradcam import GradCAM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names dynamically from dataset (fixes Bug 6 — hardcoded names)
dataset = datasets.ImageFolder(root="../Dataset/train")
class_names = dataset.classes
num_classes = len(class_names)

model = get_model(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# For EfficientNet-B0, use the last features block for GradCAM
gradcam = GradCAM(model, model.features[-1])

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Fixed Bug 4: Added ImageNet normalization to match training transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


def explain(cam, img_w, img_h):
    # Determine the zones relative to the output image size
    cell_h = img_h // 3
    cell_w = img_w // 3
    
    # Evaluate over the 224x224 cam dimensions, but yield original image bounding coordinates
    cam_cell_h, cam_cell_w = cam.shape[0] // 3, cam.shape[1] // 3
    
    grid = {
        "Top-Left (Roof/Background)": {
            "slice": cam[0:cam_cell_h, 0:cam_cell_w], 
            "bbox": (0, 0, cell_w, cell_h)
        },
        "Top-Center (Windshield/Roof)": {
            "slice": cam[0:cam_cell_h, cam_cell_w:2*cam_cell_w], 
            "bbox": (cell_w, 0, 2*cell_w, cell_h)
        },
        "Top-Right (Roof/Background)": {
            "slice": cam[0:cam_cell_h, 2*cam_cell_w:cam.shape[1]], 
            "bbox": (2*cell_w, 0, img_w, cell_h)
        },
        "Middle-Left (Headlight/Side)": {
            "slice": cam[cam_cell_h:2*cam_cell_h, 0:cam_cell_w], 
            "bbox": (0, cell_h, cell_w, 2*cell_h)
        },
        "Center (Grille/Logo/Body)": {
            "slice": cam[cam_cell_h:2*cam_cell_h, cam_cell_w:2*cam_cell_w],
            "bbox": (cell_w, cell_h, 2*cell_w, 2*cell_h)
        },
        "Middle-Right (Headlight/Side)": {
            "slice": cam[cam_cell_h:2*cam_cell_h, 2*cam_cell_w:cam.shape[1]], 
            "bbox": (2*cell_w, cell_h, img_w, 2*cell_h)
        },
        "Bottom-Left (Wheels/Bumper)": {
            "slice": cam[2*cam_cell_h:cam.shape[0], 0:cam_cell_w], 
            "bbox": (0, 2*cell_h, cell_w, img_h)
        },
        "Bottom-Center (Lower Grille/Bumper)": {
            "slice": cam[2*cam_cell_h:cam.shape[0], cam_cell_w:2*cam_cell_w], 
            "bbox": (cell_w, 2*cell_h, 2*cell_w, img_h)
        },
        "Bottom-Right (Wheels/Bumper)": {
            "slice": cam[2*cam_cell_h:cam.shape[0], 2*cam_cell_w:cam.shape[1]], 
            "bbox": (2*cell_w, 2*cell_h, img_w, img_h)
        }
    }
    
    region_scores = {name: np.mean(data["slice"]) for name, data in grid.items()}
    sorted_regions = sorted(region_scores.items(), key=lambda item: item[1], reverse=True)
    
    top_2 = sorted_regions[:2]
    
    results = []
    for name, score in top_2:
        results.append((name, grid[name]["bbox"]))
        
    return results


def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    # gradcam generates a 224x224 cam
    cam = gradcam.generate(input_tensor, pred_class.item())

    # We read the original image to draw on it instead of crushing it down to 224x224
    orig_img = cv2.imread(image_path)
    orig_h, orig_w = orig_img.shape[:2]

    # Resize CAM to original image dimensions for a crisp overlay
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    
    # Get explanations and bounding boxes scaled to original image
    explanations = explain(cam, orig_w, orig_h)

    # Superimpose heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    # Draw bounding boxes
    reasons = []
    for rank, (name, (x1, y1, x2, y2)) in enumerate(explanations):
        reasons.append(name)
        
        # Draw neon green box
        thickness = max(2, orig_w // 150)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        
        # Add a label background
        text = f"#{rank+1} Key Region"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, orig_w / 800)
        font_thickness = max(1, orig_w // 400)
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(overlay, (x1, y1), (x1 + text_w + 10, y1 + text_h + 10), (0, 255, 0), -1)
        
        # Add text
        cv2.putText(overlay, text, (x1 + 5, y1 + text_h + 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    output_path = "gradcam_result.jpg"
    cv2.imwrite(output_path, overlay)

    print(f"\nPrediction: {class_names[pred_class.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    print(f"Primary Features Detected: {reasons}")
    print(f"✅ Visualization saved with bounding boxes to {output_path}!")


if __name__ == "__main__":
    predict("test.jpg")