import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import time
import matplotlib.pyplot as plt
import os

# Load the Faster RCNN model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Ensure results directory exists
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Preprocess image function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0), image

# Object detection function
def detect_objects(model, image_tensor, threshold=0.5):
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)
    inference_time = time.time() - start_time
    
    filtered_predictions = []
    for idx, score in enumerate(predictions[0]['scores']):
        if score > threshold:
            filtered_predictions.append({
                'bbox': predictions[0]['boxes'][idx].cpu().numpy(),
                'label': predictions[0]['labels'][idx].item(),
                'score': score.item()
            })
    return filtered_predictions, inference_time

# Function to get human-readable labels (you may need to adjust the mapping)
def get_label_name(label_id):
    # Example mapping: Customize this according to your model's classes
    label_map = {
        1: "Person",
        2: "Bicycle",
        3: "Car",
        4: "Motorcycle",
        5: "Airplane",
        6: "Bus",
        7: "Train",
        8: "Truck",
        9: "Boat",
        10: "Traffic Light",
        # Add all the class labels based on COCO dataset or your dataset
    }
    return label_map.get(label_id, "Unknown")

# Display and save results function
def display_and_save_results(image, predictions, image_name):
    draw = ImageDraw.Draw(image)
    
    # Load a font (use a path to a .ttf file or use default font)
    font_size = 24  # Increase font size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Make sure this font is available on your system
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if custom font is not available

    for pred in predictions:
        bbox = pred['bbox']
        label = get_label_name(pred['label'])  # Get human-readable label
        score = pred['score']
        
        # Draw bounding box
        draw.rectangle(bbox, outline="red", width=3)
        
        # Add label and score
        text = f"{label}: {score:.2f}"
        draw.text((bbox[0], bbox[1] - 10), text, fill="red", font=font)  # Use the larger font
    
    # Save and display image
    result_path = os.path.join(results_folder, f"{image_name}_detected.jpg")
    image.save(result_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Main loop for processing images
results = []
image_folder = 'images/'  # Folder containing your images
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image_tensor, original_image = preprocess_image(image_path)
    predictions, inference_time = detect_objects(model, image_tensor, threshold=0.7)
    display_and_save_results(original_image, predictions, image_name)
    
    # Store results
    result = {
        "image_name": image_name,
        "inference_time": inference_time,
        "predictions": predictions  # Can calculate IoU if ground truth is available
    }
    results.append(result)
