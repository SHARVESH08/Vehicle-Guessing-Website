import os
import io
import base64
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms, datasets
from model import get_model
from gradcam import GradCAM

app = Flask(__name__)
CORS(app)

# ----------------------------------------
# 1. Initialize Global Model & Variables
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model for Web API on:", device)

# Hardcoded to bypass the need for massive Dataset folders on cloud servers
class_names = [
    '1966_Dodge_Charger', '1969_Dino_246_GT', '1970_Lincoln_Continental_fourdoor', 
    '1971_Maserati_Bora', '1980_Ferrari_Mondial_8', '1988_Volkswagen_Passat', 
    '1993_Jeep_Grand_Cherokee_ZJ', '2000_Toyota_Corolla_Sedan', '2001_Nissan_XTrail', 
    '2002_Chevrolet_Corvette_C5_Z06'
]
num_classes = len(class_names)

model = get_model(num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

gradcam = GradCAM(model, model.features[-1])

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

def explain(cam, img_w, img_h):
    cell_h = img_h // 3
    cell_w = img_w // 3
    cam_cell_h, cam_cell_w = cam.shape[0] // 3, cam.shape[1] // 3
    
    grid = {
        "Top-Left (Roof/Background)": {"slice": cam[0:cam_cell_h, 0:cam_cell_w], "bbox": (0, 0, cell_w, cell_h)},
        "Top-Center (Windshield/Roof)": {"slice": cam[0:cam_cell_h, cam_cell_w:2*cam_cell_w], "bbox": (cell_w, 0, 2*cell_w, cell_h)},
        "Top-Right (Roof/Background)": {"slice": cam[0:cam_cell_h, 2*cam_cell_w:cam.shape[1]], "bbox": (2*cell_w, 0, img_w, cell_h)},
        "Middle-Left (Side Profile)": {"slice": cam[cam_cell_h:2*cam_cell_h, 0:cam_cell_w], "bbox": (0, cell_h, cell_w, 2*cell_h)},
        "Center (Grille/Logo/Body)": {"slice": cam[cam_cell_h:2*cam_cell_h, cam_cell_w:2*cam_cell_w], "bbox": (cell_w, cell_h, 2*cell_w, 2*cell_h)},
        "Middle-Right (Side Profile)": {"slice": cam[cam_cell_h:2*cam_cell_h, 2*cam_cell_w:cam.shape[1]], "bbox": (2*cell_w, cell_h, img_w, 2*cell_h)},
        "Bottom-Left (Wheels/Bumper)": {"slice": cam[2*cam_cell_h:cam.shape[0], 0:cam_cell_w], "bbox": (0, 2*cell_h, cell_w, img_h)},
        "Bottom-Center (Lower Grille/Bumper)": {"slice": cam[2*cam_cell_h:cam.shape[0], cam_cell_w:2*cam_cell_w], "bbox": (cell_w, 2*cell_h, 2*cell_w, img_h)},
        "Bottom-Right (Wheels/Bumper)": {"slice": cam[2*cam_cell_h:cam.shape[0], 2*cam_cell_w:cam.shape[1]], "bbox": (2*cell_w, 2*cell_h, img_w, img_h)}
    }
    
    region_scores = {name: np.mean(data["slice"]) for name, data in grid.items()}
    sorted_regions = sorted(region_scores.items(), key=lambda item: item[1], reverse=True)
    return [(name, grid[name]["bbox"]) for name, _ in sorted_regions[:2]]

# ----------------------------------------
# 2. Flask Routes
# ----------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static_src/<path:filename>')
def serve_static(filename):
    # Serve images from src to frontend securely
    return send_from_directory('.', filename)

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    import json
    try:
        with open('metrics.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "metrics.json not found"}), 404

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    img_bytes = file.read()
    
    # Process original image for display/cv2
    nparr = np.frombuffer(img_bytes, np.uint8)
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_h, orig_w = orig_img.shape[:2]
    
    # Process for PyTorch
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    cam = gradcam.generate(input_tensor, pred_class.item())
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    explanations = explain(cam, orig_w, orig_h)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    reasons = []
    for rank, (name, (x1, y1, x2, y2)) in enumerate(explanations):
        reasons.append(name)
        thickness = max(2, orig_w // 150)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        text = f"#{rank+1} Region"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, orig_w / 800)
        font_thickness = max(1, orig_w // 400)
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(overlay, (x1, y1), (x1 + text_w + 10, y1 + text_h + 10), (0, 255, 0), -1)
        cv2.putText(overlay, text, (x1 + 5, y1 + text_h + 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', overlay)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    # Convert the prediction class into a readable string
    car_model_name = class_names[pred_class.item()].replace('_', ' ')

    return jsonify({
        "prediction": car_model_name,
        "confidence": round(confidence.item() * 100, 2),
        "reasons": reasons,
        "image_data": f"data:image/jpeg;base64,{base64_img}"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
