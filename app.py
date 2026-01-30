import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI & DATABASE KOORDINAT
# ==========================================
# Copy lagi PARKING_ROIS yang sudah fix kemarin (Y Step 50)
PARKING_ROIS = {
    # --- KOLOM 1 ---
    "A-01": [45, 103, 100, 32],
    "A-02": [45, 153, 100, 32],
    "A-03": [45, 203, 100, 32],
    "A-04": [45, 253, 100, 32],
    "A-05": [45, 303, 100, 32],
    "A-06": [45, 353, 100, 32],
    
    # --- KOLOM 3 (Contoh Sebagian) ---
    "C-01": [398, 103, 100, 32],
    "C-02": [398, 153, 100, 32],
    "C-03": [398, 203, 100, 32],
    "C-04": [398, 253, 100, 32],
}

# Load Model YOLOv11 (Hasil Training Kamu)
model = YOLO('best.pt') 

# Class Names (Sesuaikan dengan data.yaml kamu)
# 0: car, 1: empty (atau sebaliknya, cek hasil print saat run)
classNames = ['car', 'empty'] 

# Buka Video
cap = cv2.VideoCapture('carPark.mp4') 

# ==========================================
# 2. LOOPING PROGRAM
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Resize Frame biar koordinat pas (Sesuai saat kita mapping kemarin)
    # Kemarin kita mapping di resolusi bawaan atau resize? 
    # Kalau video asli besar, kita resize dulu biar ringan & koordinat pas.
    # frame = cv2.resize(frame, (1020, 720)) 

    # 2. Deteksi dengan YOLO
    results = model(frame, stream=True, verbose=False)

    # List untuk menyimpan titik tengah mobil yang terdeteksi
    car_points = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Ambil Info Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # KITA CUMA BUTUH KELAS 'CAR' (Mobil)
            # Logikanya: Kalau ada mobil di kotak A-01, berarti terisi.
            # Kalau tidak ada mobil, berarti kosong.
            if currentClass == 'car' and conf > 0.4:
                # Cari titik tengah mobil (Center Point)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                car_points.append((cx, cy)) # Simpan titik tengah

                # Gambar kotak mobil (Opsional - buat debug aja)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

    # 3. LOGIKA PENCOCOKAN (MAPPING MATCHING)
    counter_empty = 0
    
    for slot_name, (x, y, w, h) in PARKING_ROIS.items():
        # Status Awal: Anggap KOSONG (Hijau)
        is_occupied = False
        color = (0, 255, 0) # Hijau
        thickness = 2

        # Cek apakah ada Titik Tengah Mobil di dalam Kotak ini?
        for (cx, cy) in car_points:
            # Rumus Matematika: Apakah cx, cy ada di dalam rentang kotak x,y,w,h?
            if x < cx < x + w and y < cy < y + h:
                is_occupied = True
                break # Ketemu satu mobil cukup, stop loop
        
        # Jika Terisi, Ubah Warna jadi MERAH
        if is_occupied:
            color = (0, 0, 255) # Merah
            thickness = 2
        else:
            counter_empty += 1 # Hitung slot kosong

        # 4. GAMBAR KOTAK & NAMA BLOK
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Buat label nama background
        cvzone.putTextRect(frame, slot_name, (x, y - 5), scale=0.7, thickness=1, offset=0, colorR=color)

    # Tampilkan Info Total
    cvzone.putTextRect(frame, f'Available: {counter_empty}/{len(PARKING_ROIS)}', (50, 50), scale=2, thickness=2, offset=10, colorR=(0,200,0))

    cv2.imshow("Smart Parking YOLOv11 + Mapping", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
