import os
import random
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

app = Flask(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'food101_model.pth')
NUTRITION_PATH = os.path.join(BASE_DIR, 'data', 'nutrition.csv')
IMAGE_PATH = os.path.join(BASE_DIR, 'static', 'images')

# Load model once on startup
def load_model():
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 101)
    )
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx_to_class = checkpoint['idx_to_class']
    return model, idx_to_class

model, idx_to_class = load_model()
nutrition_df = pd.read_csv(NUTRITION_PATH)

# Preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor, image

# Predict label and confidence
def predict_label(tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        top_prob, top_idx = torch.max(prob, 1)
        label = idx_to_class[top_idx.item()]
        confidence = top_prob.item()
    return label, confidence

# Get weight options per category from nutrition.csv
def get_weight_options(categories):
    weight_options = {}
    for cat in categories:
        w = nutrition_df[nutrition_df['label'].str.contains(cat, case=False)]['weight'].unique()
        weight_options[cat] = sorted(w.tolist())
    return weight_options

# Find nutrition data by label and weight
def get_nutrition(label, weight):
    result = nutrition_df[nutrition_df['label'].str.contains(label, case=False)]
    match = result[result['weight'] == float(weight)]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        'Calories': row['calories'],
        'Protein (g)': row['protein'],
        'Carbohydrates (g)': row['carbohydrates'],
        'Fat (g)': row['fats'],
        'Fiber (g)': row['fiber'],
        'Sugars (g)': row['sugars'],
        'Sodium (mg)': row['sodium']
    }

@app.route('/', methods=['GET'])
def index():
    categories = [cat for cat in os.listdir(IMAGE_PATH) if os.path.isdir(os.path.join(IMAGE_PATH, cat))]
    weight_options = get_weight_options(categories)
    return render_template('index.html', categories=categories, weight_options=weight_options)

@app.route('/predict', methods=['POST'])
def predict():
    category = request.form.get('category')
    weight = request.form.get('weight')

    categories = [cat for cat in os.listdir(IMAGE_PATH) if os.path.isdir(os.path.join(IMAGE_PATH, cat))]
    weight_options = get_weight_options(categories)

    if not category or category not in categories:
        return render_template('index.html', categories=categories, weight_options=weight_options,
                               error="Kategori tidak valid.")
    try:
        weight = float(weight)
    except:
        return render_template('index.html', categories=categories, weight_options=weight_options,
                               error="Berat tidak valid.")

    cat_dir = os.path.join(IMAGE_PATH, category)
    image_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return render_template('index.html', categories=categories, weight_options=weight_options,
                               error="Tidak ada gambar di kategori ini.")

    random_img_name = random.choice(image_files)
    image_path = os.path.join(cat_dir, random_img_name)

    tensor, img = preprocess_image(image_path)
    label, confidence = predict_label(tensor)
    nutrition = get_nutrition(label, weight)

    image_display = f"/static/images/{category}/{random_img_name}"

    result = {
        'label': label,
        'confidence': confidence,
        'nutrition': nutrition
    }

    return render_template('result.html', result=result, image_display=image_display, weight=weight)

if __name__ == '__main__':
    app.run(debug=True)
