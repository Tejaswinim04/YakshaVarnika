from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)  # changed _name_ to __name__
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists (optional but recommended)
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your trained model
model = tf.keras.models.load_model('yakshagana_cnn_model.h5')

# IMPORTANT: Update this list to match your labels
class_labels = ['Chanda_Munda', 'Devi', 'Krishna', 'Mahishasura', 'Stree_Vesha']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')   # Homepage only

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = Image.open(filepath)
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)
            class_idx = np.argmax(pred)
            prediction = class_labels[class_idx]

    return render_template('upload.html', prediction=prediction, filename=filename)


if __name__ == '__main__':  # changed _name_ and _main_ to __name__ and __main__
    app.run(debug=True)       # removed duplicate run call

