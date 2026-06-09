from flask import Flask, render_template, Response, jsonify, request, send_from_directory, session, redirect, url_for
from functools import wraps
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
import queue
import csv
from io import StringIO
from flask import make_response
import random
import json  # WAJIB UNTUK KONVERSI ARRAY KE TEKS DATABASE

app = Flask(__name__)
app.secret_key = 'vision_ai_kunci_rahasia_super_aman' 

# ==========================================
# INISIALISASI DATABASE & FOLDER
# ==========================================
DATASET_DIR = "dataset_wajah"
if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)

def get_db_connection():
    return sqlite3.connect('attendance.db', check_same_thread=False, timeout=15.0)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # OPTIMASI SKALABILITAS: Tambahkan kolom 'encoding TEXT' untuk menyimpan otak AI
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT UNIQUE, waktu_daftar DATETIME DEFAULT CURRENT_TIMESTAMP, encoding TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT, tanggal TEXT, waktu TEXT, status TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS admins (username TEXT PRIMARY KEY, password TEXT)''')
    
    cursor.execute("SELECT * FROM admins WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO admins (username, password) VALUES ('admin', 'admin123')")
    conn.commit()
    conn.close()

init_db()

# ==========================================
# DEKORATOR KEAMANAN
# ==========================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==========================================
# VARIABEL GLOBAL & STATE AI
# ==========================================
known_face_encodings, known_face_names = [], []
wajah_sudah_absen = {}
tanggal_terakhir_absen = datetime.now().date()

FACE_MATCH_TOLERANCE = 0.40
EAR_THRESHOLD = 0.20 
MAR_THRESHOLD = 0.35 

output_frame = None          
raw_frame_for_register = None 
frame_lock = threading.Lock() 
ai_lock = threading.Lock() 
latest_scan_event = {}

# ==========================================
# DEDICATED AUDIO WORKER
# ==========================================
audio_queue = queue.Queue()

def audio_worker():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'indonesia' in voice.name.lower() or 'id' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 150)
        while True:
            teks = audio_queue.get()
            if teks is None: break
            try:
                engine.say(teks)
                engine.runAndWait()
            except: pass
            audio_queue.task_done()
    except: pass

threading.Thread(target=audio_worker, daemon=True).start()

def ucapkan_pesan(teks):
    audio_queue.put(teks)

# ==========================================
# FUNGSI LIVENESS DETECTION & UTILITAS
# ==========================================
def load_registered_faces():
    """ 
    OPTIMASI SKALABILITAS: Hanya baca teks JSON dari Database. 
    Tidak ada lagi proses baca file .jpg satu per satu saat server menyala!
    """
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT nama, encoding FROM users WHERE encoding IS NOT NULL")
    for row in cursor.fetchall():
        nama = row[0]
        encoding_str = row[1]
        try:
            # Ubah teks JSON kembali menjadi bentuk Array yang dipahami AI
            encoding_array = np.array(json.loads(encoding_str))
            known_face_encodings.append(encoding_array)
            known_face_names.append(nama)
        except Exception as e:
            print(f"[ERROR] Gagal memuat data wajah {nama} dari database.")
            
    conn.close()

def muat_absensi_hari_ini():
    global wajah_sudah_absen, tanggal_terakhir_absen
    wajah_sudah_absen.clear()
    tanggal_terakhir_absen = datetime.now().date()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT nama, waktu FROM attendance WHERE tanggal=?", (tanggal_terakhir_absen.strftime("%Y-%m-%d"),))
        for row in cursor.fetchall():
            wajah_sudah_absen[row[0]] = row[1]
        conn.close()
    except: pass

load_registered_faces()
muat_absensi_hari_ini()

def hitung_jarak(p1, p2): return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def eye_aspect_ratio(eye):
    A = hitung_jarak(eye[1], eye[5])
    B = hitung_jarak(eye[2], eye[4])
    C = hitung_jarak(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def mouth_aspect_ratio(top_lip, bottom_lip):
    w = hitung_jarak(top_lip[0], top_lip[6])
    h1 = hitung_jarak(top_lip[2], bottom_lip[4])
    h2 = hitung_jarak(top_lip[3], bottom_lip[3])
    h3 = hitung_jarak(top_lip[4], bottom_lip[2])
    h_avg = (h1 + h2 + h3) / 3.0
    return h_avg / w if w != 0 else 0

def catat_kehadiran_db(nama):
    try:
        now = datetime.now()
        tanggal, jam = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM attendance WHERE nama=? AND tanggal=?", (nama, tanggal))
        if cursor.fetchone():
            conn.close()
            return None, None
        batas_waktu = datetime.strptime("08:30:00", "%H:%M:%S").time()
        status = "Tepat Waktu" if now.time() <= batas_waktu else "Terlambat"
        cursor.execute("INSERT INTO attendance (nama, tanggal, waktu, status) VALUES (?, ?, ?, ?)", (nama, tanggal, jam, status))
        conn.commit()
        conn.close()
        return jam, status
    except: return None, None

# ==========================================
# WORKER KAMERA BACKGROUND 
# ==========================================
def camera_worker():
    global output_frame, raw_frame_for_register, wajah_sudah_absen, tanggal_terakhir_absen, latest_scan_event
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    process_this_frame = True
    face_locations, face_encodings, face_landmarks_list = [], [], []
    active_challenges = {}

    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.1)
                continue

            raw_frame = frame.copy()
            tgl_sekarang = datetime.now().date()
            if tgl_sekarang > tanggal_terakhir_absen:
                wajah_sudah_absen.clear()
                tanggal_terakhir_absen = tgl_sekarang

            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                with ai_lock:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    if len(face_locations) > 0:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        try: face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                        except: face_landmarks_list = [{}] * len(face_locations)
                    else:
                        face_encodings, face_landmarks_list = [], []
            process_this_frame = not process_this_frame

            for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                name = "Unknown"
                color = (0, 0, 255)
                display_text = "Wajah Tidak Dikenali"

                ear, mar = 1.0, 0.0

                if 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
                    ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0
                if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
                    mar = mouth_aspect_ratio(face_landmarks['top_lip'], face_landmarks['bottom_lip'])

                if known_face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if face_distances[best_match_index] < FACE_MATCH_TOLERANCE:
                            name = known_face_names[best_match_index]
                            
                            if name in wajah_sudah_absen:
                                color, display_text = (255, 255, 0), f"{name} (Selesai)"
                                if name in active_challenges: del active_challenges[name]
                            else:
                                if name not in active_challenges:
                                    active_challenges[name] = {
                                        "aksi": random.choice(["KEDIPKAN MATA", "BUKA MULUT"]),
                                        "sukses_frames": 0,
                                        "tutup_sebelumnya": False
                                    }
                                
                                tantangan = active_challenges[name]["aksi"]
                                color = (0, 255, 255) 
                                display_text = f"{name} | {tantangan}!"
                                liveness_passed = False

                                if tantangan == "KEDIPKAN MATA":
                                    if ear < EAR_THRESHOLD:
                                        active_challenges[name]["sukses_frames"] += 1
                                    else:
                                        if active_challenges[name]["sukses_frames"] >= 2:
                                            active_challenges[name]["tutup_sebelumnya"] = True
                                        active_challenges[name]["sukses_frames"] = 0
                                        
                                    if active_challenges[name]["tutup_sebelumnya"] and ear >= EAR_THRESHOLD:
                                        liveness_passed = True

                                elif tantangan == "BUKA MULUT":
                                    if mar > MAR_THRESHOLD:
                                        active_challenges[name]["sukses_frames"] += 1
                                        if active_challenges[name]["sukses_frames"] >= 3:
                                            liveness_passed = True
                                    else:
                                        active_challenges[name]["sukses_frames"] = 0

                                if liveness_passed:
                                    color, display_text = (0, 255, 0), f"{name} Diverifikasi"
                                    jam_masuk, status = catat_kehadiran_db(name)
                                    if jam_masuk: 
                                        wajah_sudah_absen[name] = jam_masuk
                                        ucapkan_pesan(f"Presensi {name}, direkam.")
                                        with frame_lock:
                                            latest_scan_event = {"nama": name, "waktu": jam_masuk, "status": status, "timestamp": time.time()}
                                    del active_challenges[name]

                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 40), color, cv2.FILLED)
                cv2.putText(frame, display_text, (left + 8, bottom + 26), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255), 1)

            with frame_lock:
                output_frame = frame.copy()
                raw_frame_for_register = raw_frame.copy()
            time.sleep(0.01)
        except: time.sleep(0.5)

t = threading.Thread(target=camera_worker, daemon=True)
t.start()

# ==========================================
# FLASK ROUTES
# ==========================================
def generate_web_stream():
    global output_frame
    while True:
        frame_copy = None
        with frame_lock:
            if output_frame is not None: frame_copy = output_frame.copy()
        if frame_copy is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame_copy)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admins WHERE username=? AND password=?", (username, password))
        admin = cursor.fetchone()
        conn.close()
        
        if admin:
            session['logged_in'] = True
            session['username'] = username
            return jsonify({"status": "success", "message": "Login berhasil!"})
        else:
            return jsonify({"status": "error", "message": "Username atau Password salah!"})
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
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
        
        employee_status = []
        for u in users:
            nama = u[0]
            if nama in wajah_sudah_absen:
                employee_status.append({"nama": nama, "status": "Hadir", "waktu": wajah_sudah_absen[nama]})
            else:
                employee_status.append({"nama": nama, "status": "Belum Hadir", "waktu": "-"})
        
        return render_template('dashboard.html', data=data[:10], users=users, hadir=hadir_hari_ini, tepat=tepat_waktu, telat=terlambat, employees=employee_status)
    except Exception as e:
        return f"Terjadi kesalahan saat memuat dashboard: {e}"

@app.route('/export_csv')
@login_required
def export_csv():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, nama, tanggal, waktu, status FROM attendance ORDER BY id DESC")
        data = cursor.fetchall()
        conn.close()

        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['ID', 'Nama Entitas', 'Tanggal', 'Jam Masuk', 'Status Kehadiran'])
        cw.writerows(data)

        output = make_response(si.getvalue())
        nama_file = f"Laporan_Kehadiran_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output.headers["Content-Disposition"] = f"attachment; filename={nama_file}"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        return f"Gagal mengekspor data: {e}"

@app.route('/scanner')
def scanner():
    return render_template('scanner.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_attendance')
def stream_attendance():
    def event_stream():
        last_sent_time = 0
        while True:
            with frame_lock:
                current_event = latest_scan_event.copy() if latest_scan_event else None
            
            if current_event and current_event.get("timestamp", 0) > last_sent_time:
                last_sent_time = current_event["timestamp"]
                yield f"data: {json.dumps(current_event)}\n\n"
            
            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/dataset/<path:filename>')
@login_required
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/current_frame')
@login_required
def current_frame():
    global output_frame
    frame_copy = None
    with frame_lock:
        if output_frame is not None: frame_copy = output_frame.copy()
    if frame_copy is None: return "", 204
    ret, buffer = cv2.imencode('.jpg', frame_copy)
    if not ret: return "", 204
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/register_face', methods=['POST'])
@login_required
def register_face():
    global raw_frame_for_register, known_face_encodings, known_face_names
    name = request.form.get('name')
    
    with frame_lock:
        if raw_frame_for_register is None: return jsonify({"status": "error", "message": "Kamera sedang memuat, coba sebentar lagi!"})
        frame_to_save = raw_frame_for_register.copy()
        
    if not name: return jsonify({"status": "error", "message": "Nama tidak boleh kosong!"})
    
    try:
        # OPTIMASI SKALABILITAS: Hitung encoding dan ubah ke JSON sebelum menyimpan ke DB
        rgb_frame = cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            return jsonify({"status": "error", "message": "Wajah tidak terdeteksi! Ulangi pengambilan foto."})
            
        new_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        encoding_json = json.dumps(new_encoding.tolist())
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE nama=?", (name,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"status": "error", "message": f"Wajah atas nama '{name}' sudah ada!"})
            
        # Simpan file JPG (hanya untuk ditampilkan di UI dashboard)
        filepath = os.path.join(DATASET_DIR, f"{name}.jpg")
        cv2.imwrite(filepath, frame_to_save)
        
        # Simpan nama DAN otak AI (encoding JSON) ke SQLite
        cursor.execute("INSERT INTO users (nama, encoding) VALUES (?, ?)", (name, encoding_json))
        conn.commit()
        conn.close()
            
        # Update memori berjalan
        known_face_encodings.append(new_encoding)
        known_face_names.append(name)
        
        return jsonify({"status": "success", "message": f"Wajah '{name}' berhasil didaftarkan!"})
    except Exception as e:
        print("[ERROR Registrasi]:", e)
        return jsonify({"status": "error", "message": "Terjadi kesalahan internal server!"})

@app.route('/delete_face', methods=['POST'])
@login_required
def delete_face():
    global known_face_encodings, known_face_names
    name = request.form.get('name')
    filepath = os.path.join(DATASET_DIR, f"{name}.jpg")
    if os.path.exists(filepath): os.remove(filepath)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE nama=?", (name,))
        conn.commit()
        conn.close()
        
        if name in known_face_names:
            idx = known_face_names.index(name)
            known_face_names.pop(idx)
            known_face_encodings.pop(idx)
            
        return jsonify({"status": "success", "message": f"Data '{name}' berhasil dihapus!"})
    except: return jsonify({"status": "error", "message": "Gagal menghapus data dari database!"})

if __name__ == '__main__':
    app.run(debug=False, port=5000)