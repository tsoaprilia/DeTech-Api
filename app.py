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
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Gambar tidak ditemukan'}), 400

    try:
        # 1. Jalankan Deteksi
        results = model.predict(source=image_path, conf=0.25, iou=0.45, save=False)
        
        # 2. Persiapan Path & Gambar Asli
        img_orig = cv2.imread(image_path)
        h_orig, w_orig, _ = img_orig.shape
        dir_name = os.path.dirname(image_path)
        # Ambil nama file tanpa ekstensi (misal: 'gigi' bukan 'gigi.jpg')
        base_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
        
        detections = []
        
        
       # Di dalam app.py, ganti bagian loop cropping:
        for box in results[0].boxes:
            fdi_number = model.names[int(box.cls)]
            conf = round(float(box.conf) * 100, 2)
            coords = box.xyxy[0].tolist() 
            x1, y1, x2, y2 = map(int, coords)
            
            # --- PERBAIKAN: PADDING LEBIH KECIL AGAR FOKUS ---
            padding = 20 # Kecilkan dari 60 ke 20 agar tidak bocor ke gigi sebelah
            
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w_orig, x2 + padding)
            py2 = min(h_orig, y2 + padding)
            
            crop_img = img_orig[py1:py2, px1:px2]
            
            if crop_img.size > 0:
                crop_filename = f"crop_{fdi_number}_{base_name_no_ext}.jpg"
                cv2.imwrite(os.path.join(dir_name, crop_filename), crop_img)
                
                detections.append({
                    'fdi': str(fdi_number), # Pastikan dikirim sebagai string
                    'confidence': conf,
                    'crop_image': crop_filename
                })

        # 4. Plot Bounding Box Utama (Untuk sisi kanan UI)
        res_plotted = results[0].plot()
        result_filename = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(dir_name, result_filename), res_plotted)

        return jsonify({
            'status': 'success',
            'total': len(detections),
            'results': detections,
            'result_image': result_filename
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Jalankan Flask di port 5000
    app.run(port=5000, debug=True)
