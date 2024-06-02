import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS module
import pandas as pd
from sklearn.linear_model import LinearRegression
import torchvision
import torch
import PIL.Image as Image
from werkzeug.utils import secure_filename
from torchvision import  transforms

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
# Set the path for templates directory
template_dir = os.path.abspath(os.path.dirname(__file__))
app.template_folder = os.path.join(template_dir)
# print(app.template_folder)

# Load the machine learning model
model = torch.load('best_rgmodel.pth')
# df = pd.read_csv("insurance.csv")  # Make sure to have your CSV file in the same directory
# X = df[['age', 'sex', 'bmi', 'children', 'smoker']]
# y = df['charges']
# model.fit(X, y)

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify(model, image_transforms, image_path, classes):
  model = model.eval()
  image = Image.open(image_path)
  image = image_transforms(image).float()
  image = image.unsqueeze(0)

  output = model(image)
  _, predicted = torch.max(output.data, 1)

  print(classes[predicted.item()])
  return classes[predicted.item()]

classes = [
    "defective",
    'good'
]

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400
        file = request.files['image']

        # If the user does not select a file, the browser might
        # submit an empty file without a filename.
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file to a temporary location
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Call the classify function with the file path
        prediction = classify(model, test_transform, file_path, classes)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

   

if __name__ == '__main__':
    app.run(debug=True)
