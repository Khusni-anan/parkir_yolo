import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN & MODEL
# ==========================================
st.set_page_config(page_title="Tes Deteksi Parkir (Simple)", layout="wide")

@st.cache_resource
def load_model():
    # Pastikan file best.pt ada satu folder dengan app.py
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model 'best.pt'. Error: {e}")
    st.stop()

# Ambil nama kelas langsung dari model (biar tidak salah urutan)
# Biasanya: {0: 'car', 1: 'free'} atau sebaliknya
class_names = model.names
st.sidebar.success(f"Model berhasil dimuat! Kelas: {class_names}")

# ==========================================
# 2. UPLOAD FILE
# ==========================================
st.title("🚗 Tes Murni Deteksi YOLO (Tanpa Mapping)")
st.write("Upload video/gambar untuk melihat apakah model bisa membedakan **Mobil** vs **Kosong**.")

uploaded_file = st.sidebar.file_uploader("Upload File Disini", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])

# ==========================================
# 3. FUNGSI GAMBAR HASIL
# ==========================================
def process_frame(frame, conf_threshold, iou_threshold):
    # Run YOLO
    # imgsz=1280 agar deteksi lebih tajam untuk objek kecil/jauh
    results = model(frame, imgsz=1280, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    # Hitung Statistik
    count_car = 0
    count_free = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Ambil koordinat & kelas
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label_name = class_names[cls].lower() # car / free
            
            # Tentukan Warna & Label
            # Mobil = Merah, Free = Hijau
            if 'car' in label_name:
                color = (0, 0, 255) # Merah
                count_car += 1
                label = f"Car {box.conf[0]:.2f}"
            elif 'free' in label_name or 'empty' in label_name:
                color = (0, 255, 0) # Hijau
                count_free += 1
                label = f"Free {box.conf[0]:.2f}"
            else:
                color = (255, 0, 0) # Biru (Lainnya)
                label = f"{label_name}"

            # Gambar Kotak & Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Background tulisan biar terbaca
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Tampilkan Total di Pojok Kiri Atas
    info_text = f"Mobil: {count_car} | Kosong: {count_free}"
    cv2.rectangle(frame, (5, 5), (350, 50), (0, 0, 0), -1) # Background hitam
    cv2.putText(frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame, count_car, count_free

# ==========================================
# 4. LOGIKA UTAMA
# ==========================================
if uploaded_file is not None:
    # Slider untuk mengatur sensitivitas (Biar user bisa main-main settingan)
    conf = st.sidebar.slider("Confidence (Keyakinan)", 0.0, 1.0, 0.25)
    iou = st.sidebar.slider("IoU (Tumpang Tindih)", 0.0, 1.0, 0.45)

    file_type = uploaded_file.name.split('.')[-1].lower()

    # --- MODE VIDEO ---
    if file_type in ['mp4', 'avi', 'mov']:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        st_metrics = st.empty()
        
        stop_btn = st.button("Stop Video")
        
        while cap.isOpened() and not stop_btn:
            success, frame = cap.read()
            if not success:
                break
            
            # Proses
            processed_frame, c_car, c_free = process_frame(frame, conf, iou)
            
            # Update Gambar
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            # Update Angka Realtime
            st_metrics.markdown(f"### 📊 Status: **{c_car}** Mobil, **{c_free}** Kosong")
            
        cap.release()

    # --- MODE GAMBAR ---
    elif file_type in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert ke BGR
        
        # Proses
        processed_frame, c_car, c_free = process_frame(frame, conf, iou)
        
        # Tampilkan
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_column_width=True)
        st.info(f"Terdeteksi: {c_car} Mobil dan {c_free} Slot Kosong.")
