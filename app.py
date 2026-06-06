from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import threading
import math
import sqlite3
import pyttsx3
import time

app = Flask(__name__)

# ==========================================
# INISIALISASI DATABASE & FOLDER
# ==========================================
DATASET_DIR = "dataset_wajah"
if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)

def get_db_connection():
    return sqlite3.connect('attendance.db', check_same_thread=False)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT UNIQUE, waktu_daftar DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT, tanggal TEXT, waktu TEXT, status TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# VARIABEL GLOBAL & STATE AI
# ==========================================
known_face_encodings, known_face_names = [], []
wajah_sudah_absen = {}
tanggal_terakhir_absen = datetime.now().date()
EAR_THRESHOLD = 0.22

# Kunci Arsitektur Baru: Variabel Global untuk menampung frame
output_frame = None          # Frame berisi kotak deteksi (untuk ditampilkan)
raw_frame_for_register = None # Frame bersih (untuk disimpan saat daftar)
frame_lock = threading.Lock() # Pengaman agar web dan kamera tidak tabrakan

# ==========================================
# FUNGSI UTILITAS SISTEM
# ==========================================
def load_registered_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(DATASET_DIR, filename)
            name = os.path.splitext(filename)[0]
            
            cursor.execute("SELECT * FROM users WHERE nama=?", (name,))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO users (nama) VALUES (?)", (name,))
                conn.commit()
            
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
    conn.close()

def muat_absensi_hari_ini():
    global wajah_sudah_absen, tanggal_terakhir_absen
    wajah_sudah_absen.clear()
    tanggal_terakhir_absen = datetime.now().date()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT nama, waktu FROM attendance WHERE tanggal=?", (tanggal_terakhir_absen.strftime("%Y-%m-%d"),))
    for row in cursor.fetchall():
        wajah_sudah_absen[row[0]] = row[1]
    conn.close()

load_registered_faces()
muat_absensi_hari_ini()

def ucapkan_pesan(teks):
    def jalankan_tts():
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'indonesia' in voice.name.lower() or 'id' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 150)
            engine.say(teks)
            engine.runAndWait()
        except: pass
    threading.Thread(target=jalankan_tts, daemon=True).start()

def hitung_jarak(p1, p2): return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
def eye_aspect_ratio(eye):
    A = hitung_jarak(eye[1], eye[5])
    B = hitung_jarak(eye[2], eye[4])
    C = hitung_jarak(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def catat_kehadiran_db(nama):
    now = datetime.now()
    tanggal, jam = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE nama=? AND tanggal=?", (nama, tanggal))
    if cursor.fetchone():
        conn.close()
        return None, None
        
    batas_waktu = datetime.strptime("08:30:00", "%H:%M:%S").time()
    status = "Tepat Waktu" if now.time() <= batas_waktu else "Terlambat"
    
    cursor.execute("INSERT INTO attendance (nama, tanggal, waktu, status) VALUES (?, ?, ?, ?)", (nama, tanggal, jam, status))
    conn.commit()
    conn.close()
    return jam, status

# ==========================================
# WORKER KAMERA BACKGROUND (ANTI-ERROR)
# ==========================================
def camera_worker():
    global output_frame, raw_frame_for_register, wajah_sudah_absen, tanggal_terakhir_absen
    
    print("[INFO] Memulai kamera utama...")
    cap = cv2.VideoCapture(0)
    mata_tertutup_sebelumnya = False
    
    # Teknik "Frame Skipping" agar AI tidak ngelag
    process_this_frame = True
    face_locations = []
    face_encodings = []
    face_landmarks_list = []

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        raw_frame = frame.copy()
        
        tgl_sekarang = datetime.now().date()
        if tgl_sekarang > tanggal_terakhir_absen:
            wajah_sudah_absen.clear()
            tanggal_terakhir_absen = tgl_sekarang

        # Hanya jalankan AI setiap selang 1 frame agar super ringan
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            try: 
                face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            except: 
                face_landmarks_list = [{}] * len(face_locations)
                
        process_this_frame = not process_this_frame

        # Gambar kotak di wajah
        for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            name, color, display_text = "Unknown", (0, 0, 255), "Tidak Dikenal"
            ear = 1.0

            if 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
                ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0

            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < 0.45:
                        name = known_face_names[best_match_index]
                        if name in wajah_sudah_absen:
                            color, display_text = (255, 255, 0), f"{name} (Sudah Absen)"
                        else:
                            color, display_text = (0, 255, 0), f"Wajah: {name}"
                            if ear < EAR_THRESHOLD:
                                mata_tertutup_sebelumnya = True
                            elif ear >= EAR_THRESHOLD and mata_tertutup_sebelumnya:
                                mata_tertutup_sebelumnya = False
                                if name not in wajah_sudah_absen:
                                    jam_masuk, status = catat_kehadiran_db(name)
                                    if jam_masuk: 
                                        wajah_sudah_absen[name] = jam_masuk
                                        ucapkan_pesan(f"Presensi {name}, direkam.")
                            
                            if not mata_tertutup_sebelumnya and name not in wajah_sudah_absen:
                                display_text = f"{name} - Berkedip utk Absen"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
            cv2.putText(frame, display_text, (left + 5, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Simpan hasil akhir ke variabel global secara aman
        with frame_lock:
            output_frame = frame.copy()
            raw_frame_for_register = raw_frame.copy()

# Mulai worker kamera di latar belakang saat server hidup
t = threading.Thread(target=camera_worker, daemon=True)
t.start()

# ==========================================
# FLASK ROUTES
# ==========================================
def generate_web_stream():
    """ Fungsi ini HANYA MENGAMBIL gambar yang sudah jadi, tidak menyentuh kamera """
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03) # 30 FPS untuk web browser

@app.route('/')
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT nama, tanggal, waktu, status FROM attendance ORDER BY id DESC")
    data = cursor.fetchall()
    cursor.execute("SELECT nama, waktu_daftar FROM users ORDER BY id DESC")
    users = cursor.fetchall()
    conn.close()
    
    tgl_hari_ini = datetime.now().strftime("%Y-%m-%d")
    hadir_hari_ini = sum(1 for row in data if row[1] == tgl_hari_ini)
    tepat_waktu = sum(1 for row in data if row[1] == tgl_hari_ini and row[3] == 'Tepat Waktu')
    terlambat = sum(1 for row in data if row[1] == tgl_hari_ini and row[3] == 'Terlambat')
    
    return render_template('dashboard.html', data=data, users=users, hadir=hadir_hari_ini, tepat=tepat_waktu, telat=terlambat)

@app.route('/scanner')
def scanner():
    return render_template('scanner.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/register_face', methods=['POST'])
def register_face():
    global raw_frame_for_register
    name = request.form.get('name')
    
    with frame_lock:
        if raw_frame_for_register is None:
            return jsonify({"status": "error", "message": "Kamera sedang memuat, coba sebentar lagi!"})
        frame_to_save = raw_frame_for_register.copy()
    
    if not name:
        return jsonify({"status": "error", "message": "Nama tidak boleh kosong!"})
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE nama=?", (name,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"status": "error", "message": f"Wajah atas nama '{name}' sudah ada!"})
    
    filepath = os.path.join(DATASET_DIR, f"{name}.jpg")
    cv2.imwrite(filepath, frame_to_save)
    
    cursor.execute("INSERT INTO users (nama) VALUES (?)", (name,))
    conn.commit()
    conn.close()
        
    load_registered_faces()
    return jsonify({"status": "success", "message": f"Wajah '{name}' berhasil didaftarkan!"})

@app.route('/delete_face', methods=['POST'])
def delete_face():
    name = request.form.get('name')
    filepath = os.path.join(DATASET_DIR, f"{name}.jpg")
    if os.path.exists(filepath): os.remove(filepath)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE nama=?", (name,))
    conn.commit()
    conn.close()
    
    load_registered_faces()
    return jsonify({"status": "success", "message": f"Data '{name}' berhasil dihapus!"})

if __name__ == '__main__':
    # HAPUS threaded=True karena kita sudah pakai arsitektur custom thread!
    app.run(debug=False, port=5000)