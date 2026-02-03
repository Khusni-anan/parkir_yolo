import streamlit as st
import cv2
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# 0. FUNGSI BANTUAN (PENGGANTI CVZONE)
# ==========================================
# Kita pakai ini agar tidak perlu install cvzone yg bikin error di Cloud
def draw_text_rect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                   colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                   offset=10, border=None, border_color=(255, 255, 255)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)
    return img

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Smart Parking YOLO", layout="wide")

# Cache model agar tidak diload berulang-ulang (Bikin Cepat)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

# Load Model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class Names
classNames = ['car', 'empty'] 

# Database Koordinat (Sesuai punya Anda)
PARKING_ROIS = {
    # --- KOLOM 1 ---
    "A-01": [45, 103, 100, 32],
    "A-02": [45, 153, 100, 32],
    "A-03": [45, 203, 100, 32],
    "A-04": [45, 253, 100, 32],
    "A-05": [45, 303, 100, 32],
    "A-06": [45, 353, 100, 32],
    
    # --- KOLOM 3 (Contoh) ---
    "C-01": [398, 103, 100, 32],
    "C-02": [398, 153, 100, 32],
    "C-03": [398, 203, 100, 32],
    "C-04": [398, 253, 100, 32],
}

# ==========================================
# 2. TAMPILAN STREAMLIT
# ==========================================
st.title("🚗 Smart Parking Detection System")
st.caption("Menggunakan YOLOv11 & OpenCV")

col1, col2 = st.columns([3, 1])

with col2:
    st.write("### Kontrol")
    run = st.checkbox('Mulai Deteksi', value=False)
    st.info("Centang kotak di atas untuk memulai video.")
    
    # Placeholder untuk statistik (Opsional)
    status_text = st.empty()

# Placeholder untuk Video
with col1:
    stframe = st.empty()

# ==========================================
# 3. LOOPING PROGRAM
# ==========================================
if run:
    cap = cv2.VideoCapture('carPark.mp4')
    
    if not cap.isOpened():
        st.error("File 'carPark.mp4' tidak ditemukan. Pastikan sudah diupload ke GitHub.")
    
    while run:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Proses Deteksi YOLO
        results = model(frame, stream=True, verbose=False)
        car_points = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == 'car' and conf > 0.4:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    car_points.append((cx, cy))

        # Logika Pencocokan
        counter_empty = 0
        for slot_name, (x, y, w, h) in PARKING_ROIS.items():
            is_occupied = False
            color = (0, 255, 0) # Hijau (BGR untuk OpenCV)
            
            for (cx, cy) in car_points:
                if x < cx < x + w and y < cy < y + h:
                    is_occupied = True
                    break 
        
            if is_occupied:
                color = (0, 0, 255) # Merah
            else:
                counter_empty += 1

            # Gambar Kotak
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Pakai Fungsi Custom (Pengganti cvzone)
            draw_text_rect(frame, slot_name, (x, y - 5), scale=0.7, thickness=1, 
                           offset=0, colorR=color)

        # Tampilkan Info Total di Layar Video
        draw_text_rect(frame, f'Available: {counter_empty}/{len(PARKING_ROIS)}', 
                       (50, 50), scale=2, thickness=2, offset=10, colorR=(0,200,0))
        
        # Update Statistik di Sidebar juga
        status_text.markdown(f"# Tersedia: {counter_empty}")

        # KONVERSI WARNA (PENTING: OpenCV pakai BGR, Streamlit pakai RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tampilkan ke Placeholder Streamlit
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
