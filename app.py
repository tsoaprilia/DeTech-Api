from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
CORS(app)

# 1. Load model best.pt milikmu
# Pastikan file best.pt berada di folder yang sama dengan file app.py ini
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_path = data.get('image_path')
    
    # Validasi keberadaan file gambar
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Gambar tidak ditemukan'}), 400

    try:
        # 2. Jalankan Deteksi YOLOv11
        # conf=0.25 (diturunkan sedikit agar lebih sensitif mendeteksi gigi)
        # iou=0.45 untuk menangani kotak yang tumpang tindih
        results = model.predict(
            source=image_path, 
            conf=0.25, 
            iou=0.45,
            save=False
        )
        
        # 3. Plotting Bounding Box (PENTING: Ini yang membuat kotak muncul)
        # Fungsi .plot() akan merender kotak, label FDI, dan skor confidence ke dalam array gambar
        res_plotted = results[0].plot()
        
        # 4. Tentukan Path Simpan Hasil ke Folder Storage Laravel
        dir_name = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        
        # Buat nama file baru dengan awalan 'result_' agar tidak menindih file asli
        result_filename = f"result_{base_name}"
        result_path = os.path.join(dir_name, result_filename)
        
        # Simpan gambar hasil deteksi (ber-bounding box) menggunakan OpenCV
        cv2.imwrite(result_path, res_plotted)

        # 5. Susun Data deteksi untuk Tabel Verifikasi di Frontend
        detections = []
        for box in results[0].boxes:
            fdi_number = model.names[int(box.cls)]
            detections.append({
                'fdi': fdi_number,
                'confidence': round(float(box.conf) * 100, 2), # Contoh: 98.50
                'box': box.xyxy[0].tolist() 
            })

        # Kirim respon balik ke Laravel
        return jsonify({
            'status': 'success',
            'total': len(detections),
            'results': detections,
            'result_image': result_filename # Nama file ini akan ditangkap oleh React
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Jalankan Flask di port 5000
    app.run(port=5000, debug=True)
