import os
import torch
import numpy as np
from PIL import Image,UnidentifiedImageError
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Load pre-trained EfficientNet model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(img):
    try:
        img = img.convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            features = model(batch_t)
        
        return features.numpy().flatten()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None



# Load and preprocess all images in the dataset
dataset_path = r'images'
image_features = {}
for filename in os.listdir(dataset_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(dataset_path, filename)
        try:
            with Image.open(img_path) as img:
                features = extract_features(img)
                if features is not None:
                    image_features[filename] = features
        except Exception as e:
            print(f"Skipping file {filename} due to processing error: {str(e)}")

print(f"Successfully processed {len(image_features)} out of {len(os.listdir(dataset_path))} images.")

# Function to find similar images
def find_similar_images(input_features, n=3):
    similarities = {}
    for filename, features in image_features.items():
        similarity = cosine_similarity([input_features], [features])[0][0]
        similarities[filename] = similarity
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]

"""# Test with a single uploaded image
test_image_path = 'car.jpg' 
similar_images = find_similar_images(test_image_path)

print("Most similar images:")
for filename, similarity in similar_images:
    print(f"{filename}: Similarity = {similarity:.4f}")

"""




