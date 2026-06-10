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
import json

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
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT UNIQUE, waktu_daftar DATETIME DEFAULT CURRENT_TIMESTAMP, encoding TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT, tanggal TEXT, waktu TEXT, status TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS admins (username TEXT PRIMARY KEY, password TEXT)''')
    
    cursor.execute("SELECT * FROM admins WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO admins (username, password) VALUES ('admin', 'admin123')")
    conn.commit()
    conn.close()

init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session: return redirect(url_for('login'))
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
MAR_THRESHOLD = 0.45 # DIPERKETAT: Harus buka mulut lebih lebar

output_frame = None          
frame_lock = threading.Lock() 
latest_scan_event = {}

registration_request = None
registration_response = None

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
def ucapkan_pesan(teks): audio_queue.put(teks)

# ==========================================
# FUNGSI LIVENESS DETECTION & UTILITAS
# ==========================================
def load_registered_faces():
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
            encoding_array = np.array(json.loads(encoding_str))
            known_face_encodings.append(encoding_array)
            known_face_names.append(nama)
        except: pass
    conn.close()

def muat_absensi_hari_ini():
    global wajah_sudah_absen, tanggal_terakhir_absen
    wajah_sudah_absen.clear()
    tanggal_terakhir_absen = datetime.now().date()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT nama, waktu FROM attendance WHERE tanggal=?", (tanggal_terakhir_absen.strftime("%Y-%m-%d"),))
        for row in cursor.fetchall(): wajah_sudah_absen[row[0]] = row[1]
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

def detect_head_pose(face_landmarks):
    try:
        chin = face_landmarks.get('chin')
        nose = face_landmarks.get('nose_bridge')
        if not chin or not nose: return "TENGAH"
        
        left_point = chin[0]
        right_point = chin[-1]
        nose_point = nose[-1] 
        
        dist_left = hitung_jarak(left_point, nose_point)
        dist_right = hitung_jarak(right_point, nose_point)
        
        if dist_right == 0: return "TENGAH"
        ratio = dist_left / dist_right
        
        # DIPERKETAT: Pengguna harus benar-benar menoleh jauh
        if ratio < 0.50: return "KANAN"
        elif ratio > 2.00: return "KIRI"
        return "TENGAH"
    except: return "TENGAH"

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
# WORKER KAMERA BACKGROUND (ULTRA-STRICT KYC & GPU OPTIMIZED)
# ==========================================
def camera_worker():
    global output_frame, wajah_sudah_absen, tanggal_terakhir_absen, latest_scan_event
    global registration_request, registration_response
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_counter = 0
    cached_face_locations = []
    cached_colors = []
    cached_display_texts = []
    
    process_this_frame = True
    active_challenges = {}
    available_actions = ["KEDIPKAN MATA", "BUKA MULUT", "TOLEH KIRI", "TOLEH KANAN"]

    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)

            raw_frame = frame.copy()
            frame_counter += 1
            
            if registration_request is not None:
                name_to_register = registration_request
                try:
                    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    # Gunakan CNN GPU untuk registrasi agar super akurat
                    locs = face_recognition.face_locations(rgb_frame, model="cnn")
                    if len(locs) == 0:
                        registration_response = {"status": "error", "message": "Wajah tidak terdeteksi! Ulangi foto."}
                    else:
                        new_encoding = face_recognition.face_encodings(rgb_frame, locs)[0]
                        encoding_json = json.dumps(new_encoding.tolist())
                        filepath = os.path.join(DATASET_DIR, f"{name_to_register}.jpg")
                        cv2.imwrite(filepath, raw_frame)
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO users (nama, encoding) VALUES (?, ?)", (name_to_register, encoding_json))
                        conn.commit()
                        conn.close()
                        known_face_encodings.append(new_encoding)
                        known_face_names.append(name_to_register)
                        registration_response = {"status": "success", "message": f"Wajah '{name_to_register}' berhasil didaftarkan!"}
                except Exception as e:
                    registration_response = {"status": "error", "message": f"Error Sistem AI: {str(e)}"}
                
                registration_request = None 
                continue 

            tgl_sekarang = datetime.now().date()
            if tgl_sekarang > tanggal_terakhir_absen:
                wajah_sudah_absen.clear()
                tanggal_terakhir_absen = tgl_sekarang

            # OPTIMASI GPU: AI berpikir setiap 2 frame saja (15 FPS AI)
            if frame_counter % 2 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Gunakan CNN GPU untuk deteksi real-time
                face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
                if len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    try: face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                    except: face_landmarks_list = [{}] * len(face_locations)
                else:
                    face_encodings, face_landmarks_list = [], []

                cached_face_locations = face_locations
                cached_colors = []
                cached_display_texts = []

                for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                    name = "Unknown"
                    color = (0, 0, 255)
                    display_text = "Wajah Tidak Dikenali"

                    ear, mar = 1.0, 0.0
                    pose = "TENGAH"

                    if 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
                        ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0
                    if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
                        mar = mouth_aspect_ratio(face_landmarks['top_lip'], face_landmarks['bottom_lip'])
                    
                    pose = detect_head_pose(face_landmarks)

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
                                        sequence = random.sample(available_actions, 3) 
                                        active_challenges[name] = {
                                            "sequence": sequence,
                                            "current_step": 0,
                                            "sukses_frames": 0,
                                            "tutup_sebelumnya": False,
                                            "cooldown": 0 
                                        }
                                    
                                    challenge_data = active_challenges[name]
                                    curr_step_idx = challenge_data["current_step"]
                                    total_steps = len(challenge_data["sequence"])
                                    
                                    if challenge_data["cooldown"] > 0:
                                        challenge_data["cooldown"] -= 1
                                        color = (0, 255, 0)
                                        display_text = f"{name} | Lanjut..."
                                    else:
                                        tantangan = challenge_data["sequence"][curr_step_idx]
                                        color = (0, 255, 255) 
                                        display_text = f"{name} | {curr_step_idx + 1}/{total_steps}: {tantangan}!"
                                        step_passed = False

                                        # Limit frame sedikit dikurangi karena diproses tiap kelipatan 2 frame
                                        if tantangan == "KEDIPKAN MATA":
                                            if ear < EAR_THRESHOLD: 
                                                challenge_data["sukses_frames"] += 1
                                                if challenge_data["sukses_frames"] >= 2: 
                                                    challenge_data["tutup_sebelumnya"] = True
                                            else:
                                                if challenge_data["tutup_sebelumnya"] and ear > EAR_THRESHOLD + 0.02: 
                                                    step_passed = True
                                                challenge_data["sukses_frames"] = 0

                                        elif tantangan == "BUKA MULUT":
                                            if mar > MAR_THRESHOLD:
                                                challenge_data["sukses_frames"] += 1
                                                if challenge_data["sukses_frames"] >= 3: 
                                                    step_passed = True
                                            else: challenge_data["sukses_frames"] = 0
                                            
                                        elif tantangan == "TOLEH KANAN":
                                            if pose == "KANAN":
                                                challenge_data["sukses_frames"] += 1
                                                if challenge_data["sukses_frames"] >= 3: 
                                                    step_passed = True
                                            else: challenge_data["sukses_frames"] = 0
                                            
                                        elif tantangan == "TOLEH KIRI":
                                            if pose == "KIRI":
                                                challenge_data["sukses_frames"] += 1
                                                if challenge_data["sukses_frames"] >= 3: 
                                                    step_passed = True
                                            else: challenge_data["sukses_frames"] = 0

                                        if step_passed:
                                            challenge_data["current_step"] += 1
                                            challenge_data["sukses_frames"] = 0
                                            challenge_data["tutup_sebelumnya"] = False
                                            challenge_data["cooldown"] = 10 # Jeda ~0.5 detik
                                            ucapkan_pesan("Bagus")
                                            
                                            if challenge_data["current_step"] >= total_steps:
                                                color, display_text = (0, 255, 0), f"{name} Diverifikasi"
                                                jam_masuk, status = catat_kehadiran_db(name)
                                                if jam_masuk: 
                                                    wajah_sudah_absen[name] = jam_masuk
                                                    ucapkan_pesan(f"Presensi {name}, direkam.")
                                                    with frame_lock:
                                                        latest_scan_event = {"nama": name, "waktu": jam_masuk, "status": status, "timestamp": time.time()}
                                                del active_challenges[name]
                    
                    cached_colors.append(color)
                    cached_display_texts.append(display_text)

            # GAMBAR KOTAK MENGGUNAKAN SKALA HD (Dari Cache)
            for (top, right, bottom, left), color, display_text in zip(cached_face_locations, cached_colors, cached_display_texts):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 50), color, cv2.FILLED)
                cv2.putText(frame, display_text, (left + 10, bottom + 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255), 2)

            with frame_lock:
                output_frame = frame.copy()
            # Hapus timer sleep agar video tetap mulus 30 FPS
        except: time.sleep(0.1)

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
        ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    global registration_request, registration_response
    name = request.form.get('name')
    if not name: return jsonify({"status": "error", "message": "Nama tidak boleh kosong!"})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE nama=?", (name,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"status": "error", "message": f"Wajah atas nama '{name}' sudah ada!"})
        conn.close()
    except Exception as e:
        return jsonify({"status": "error", "message": "Error database!"})
        
    registration_response = None
    registration_request = name
    
    timeout = 150 
    while registration_response is None and timeout > 0:
        time.sleep(0.1)
        timeout -= 1
        
    if registration_response is None:
        registration_request = None 
        return jsonify({"status": "error", "message": "Kamera sedang sibuk, gagal memproses wajah!"})
        
    return jsonify(registration_response)

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