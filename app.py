from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

# Load model best.pt milikmu
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Gambar tidak ditemukan'}), 400

    # Jalankan Deteksi dengan parameter baru
    # conf=0.4 agar lebih akurat tapi tidak terlalu ketat
    # iou=0.4 agar kotak tidak tumpang tindih berlebihan
    results = model.predict(
        source=image_path, 
        conf=0.4, 
        iou=0.4,
        save=False # Tidak perlu simpan file gambar di server Flask
    )
    
    detections = []
    for r in results:
        for box in r.boxes:
            fdi_number = model.names[int(box.cls)]
            detections.append({
                'fdi': fdi_number,
                'confidence': round(float(box.conf) * 100, 2), # Ubah ke persen (misal: 95.5)
                'box': box.xyxy[0].tolist() 
            })

    return jsonify({
        'status': 'success',
        'total': len(detections),
        'results': detections
    })

if __name__ == '__main__':
    # Jalankan di port 5000
    app.run(port=5000, debug=True)