from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('model3.h5')

# Initialize Flask application
app = Flask(__name__)

def preprocess_image(image):
    # Resize the image to (224, 224)
    img = image.resize((224, 224))
    
    # Ensure the image has 3 color channels (convert to RGB if necessary)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Expand the dimensions to include the batch size dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('frontend.html')

# Route to handle image prediction
@app.route('/image', methods=['POST'])
def predict_image():
    if 'imageInput' in request.files:
        img_file = request.files['imageInput']
        if img_file.filename == '':
            return jsonify({'error': 'No file selected'})
        try:
            img = Image.open(img_file)
            img_array = preprocess_image(img)
            pred = model.predict(img_array)
            if pred[0][0] > 0.5:
                result = "Fire Detected"
            else:
                result = "Smoke Detected"
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': 'Error processing image: ' + str(e)})
    elif 'path' in request.form:
        img_path = request.form['path']
        try:
            img = Image.open(img_path)
            img_array = preprocess_image(img)
            pred = model.predict(img_array)
            if pred[0][0] > 0.5:
                result = "Fire Detected"
            else:
                result = "Smoke Detected"
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error': 'Error processing image: ' + str(e)})
    else:
        return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
