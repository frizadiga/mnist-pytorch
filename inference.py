#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
from classifier import SimpleCNN, TwoLayerFC

def load_model(model_path, model_type='cnn'):
    """Load a trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type.lower() == 'cnn':
        model = SimpleCNN()
    else:
        model = TwoLayerFC()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess an image for MNIST prediction"""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x if x.mean() > 0.5 else x),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def predict_digit(model, image, device):
    """Predict digit from preprocessed image"""
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_digit].item()

    return predicted_digit, confidence, probabilities[0].cpu().numpy()

def main():
    model_path = 'model.pth'

    # Load model
    try:
        model, device = load_model(model_path, model_type='cnn')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Please train the model first by running classifier.py")
        return

    # Get image path from command line or input
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to image file: ").strip()

    if image_path and image_path != "":
        try:
            # Preprocess image
            processed_image = preprocess_image(image_path)

            # Make prediction
            predicted_digit, confidence, probabilities = predict_digit(model, processed_image, device)

            print(f"Predicted digit: {predicted_digit}")
            print(f"Confidence: {confidence:.2%}")

            # Show top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            print("Top 3 predictions:")
            for i, idx in enumerate(top_3_indices):
                print(f"  {i+1}. Digit {idx}: {probabilities[idx]:.2%}")

        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("No image path provided.")

if __name__ == "__main__":
    main()
