from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULT_FOLDER'] = './static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

print(os.path.dirname(__file__))
model_path = os.path.join(os.path.dirname(__file__), 'weights/bestv8.pt')
model = YOLO(model_path)
model.conf = 0.5

def detect_on_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (640, 640))
    
    results = model(resized_img)
    result_img = results[0].plot()
    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    return output_path

def detect_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(video_path))
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        if out is None:
            height, width, _ = annotated_frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        out.write(annotated_frame)

    cap.release()
    if out:
        out.release()
    return output_path

@app.route('/')
def index():
    return render_template('index.html', result_file=None)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    if filename.lower().endswith(('.mp4')):
        output_path = detect_on_video(filepath)
        result_file = 'static/results/' + os.path.basename(output_path)
    elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        output_path = detect_on_image(filepath)
        result_file = 'static/results/' + os.path.basename(output_path)

    return render_template('index.html', result_file=result_file)

if __name__ == '__main__':
    app.run(debug=False)
