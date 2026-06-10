from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for, make_response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
import cv2
import face_recognition
import numpy as np
import os
import threading
import math
import pyttsx3
import time
import queue
import csv
from io import StringIO
import random
import platform
import json

app = Flask(__name__)
app.secret_key = 'vision_ai_kunci_rahasia_super_aman'

# ==========================================
# KONFIGURASI DATABASE MODERN (RDBMS ORM)
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance_modern.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

DATASET_DIR = "dataset_wajah"
if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)

# ==========================================
# STRUKTUR TABEL (MODELS) 
# ==========================================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nama = db.Column(db.String(100), unique=True, nullable=False)
    waktu_daftar = db.Column(db.DateTime, default=datetime.utcnow)
    encoding = db.Column(db.Text, nullable=True)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nama = db.Column(db.String(100), nullable=False)
    tanggal = db.Column(db.String(20), nullable=False)
    waktu = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), nullable=False)

class Admin(db.Model):
    __tablename__ = 'admins'
    username = db.Column(db.String(50), primary_key=True)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='hrd')

class Setting(db.Model):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(100), nullable=False)

app_settings = {}

def load_settings():
    global app_settings
    with app.app_context():
        settings = Setting.query.all()
        for s in settings: app_settings[s.key] = s.value

with app.app_context():
    db.create_all()
    if not Admin.query.filter_by(username='admin').first():
        db.session.add(Admin(username='admin', password=generate_password_hash('admin123'), role='superadmin'))
    if not Admin.query.filter_by(username='hrd').first():
        db.session.add(Admin(username='hrd', password=generate_password_hash('hrd123'), role='hrd'))
    
    default_settings = {
        'batas_waktu': '08:30:00',
        'ear_threshold': '0.20',
        'mar_threshold': '0.45',
        'spoof_threshold': '50.0'
    }
    for k, v in default_settings.items():
        if not Setting.query.filter_by(key=k).first():
            db.session.add(Setting(key=k, value=v))
    db.session.commit()
    
load_settings()

# ==========================================
# VARIABEL GLOBAL STATISTIK (DASHBOARD & AI)
# ==========================================
known_face_encodings, known_face_names = [], []
wajah_sudah_absen = {}
tanggal_terakhir_absen = datetime.now().date()
FACE_MATCH_TOLERANCE = 0.40

# Variabel Pelacak Statistik Global
total_scans = 0
total_confidence = 0.0
total_inference_time = 0.0

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
            try: engine.say(teks); engine.runAndWait()
            except: pass
            audio_queue.task_done()
    except: pass

threading.Thread(target=audio_worker, daemon=True).start()
def ucapkan_pesan(teks): audio_queue.put(teks)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def superadmin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session: return redirect(url_for('login'))
        if session.get('role') != 'superadmin': 
            return jsonify({"status": "error", "message": "Akses Ditolak!"}), 403
        return f(*args, **kwargs)
    return decorated_function

def is_spoof_texture(frame_bgr, top, right, bottom, left):
    h, w = frame_bgr.shape[:2]
    t, b = max(0, top - 20), min(h, bottom + 20)
    l, r = max(0, left - 20), min(w, right + 20)
    face_roi = frame_bgr[t:b, l:r]
    if face_roi.size == 0: return True
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = float(app_settings.get('spoof_threshold', 50.0))
    return laplacian_var < threshold

def align_and_encode_face(rgb_frame, landmarks):
    if 'left_eye' not in landmarks or 'right_eye' not in landmarks: return None
    left_center = np.array(landmarks['left_eye']).mean(axis=0).astype(int)
    right_center = np.array(landmarks['right_eye']).mean(axis=0).astype(int)
    angle = np.degrees(np.arctan2(right_center[1] - left_center[1], right_center[0] - left_center[0]))
    eyes_center = (int((left_center[0] + right_center[0]) / 2), int((left_center[1] + right_center[1]) / 2))
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_face = cv2.warpAffine(rgb_frame, M, (rgb_frame.shape[1], rgb_frame.shape[0]), flags=cv2.INTER_CUBIC)
    aligned_locs = face_recognition.face_locations(aligned_face, model="cnn")
    if aligned_locs: return face_recognition.face_encodings(aligned_face, aligned_locs)[0]
    return None

def load_registered_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()
    with app.app_context():
        users = User.query.filter(User.encoding.isnot(None)).all()
        for u in users:
            try:
                known_face_encodings.append(np.array(json.loads(u.encoding)))
                known_face_names.append(u.nama)
            except: pass

def muat_absensi_hari_ini():
    global wajah_sudah_absen, tanggal_terakhir_absen
    wajah_sudah_absen.clear()
    tanggal_terakhir_absen = datetime.now().date()
    with app.app_context():
        logs = Attendance.query.filter_by(tanggal=tanggal_terakhir_absen.strftime("%Y-%m-%d")).all()
        for log in logs: wajah_sudah_absen[log.nama] = log.waktu

load_registered_faces()
muat_absensi_hari_ini()

def hitung_jarak(p1, p2): return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
def eye_aspect_ratio(eye):
    A, B, C = hitung_jarak(eye[1], eye[5]), hitung_jarak(eye[2], eye[4]), hitung_jarak(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0
def mouth_aspect_ratio(top_lip, bottom_lip):
    w = hitung_jarak(top_lip[0], top_lip[6])
    h_avg = (hitung_jarak(top_lip[2], bottom_lip[4]) + hitung_jarak(top_lip[3], bottom_lip[3]) + hitung_jarak(top_lip[4], bottom_lip[2])) / 3.0
    return h_avg / w if w != 0 else 0
def detect_head_pose(face_landmarks):
    try:
        chin, nose = face_landmarks.get('chin'), face_landmarks.get('nose_bridge')
        if not chin or not nose: return "TENGAH"
        ratio = hitung_jarak(chin[0], nose[-1]) / hitung_jarak(chin[-1], nose[-1])
        if ratio < 0.75: return "KANAN"
        elif ratio > 1.35: return "KIRI"
        return "TENGAH"
    except: return "TENGAH"

def catat_kehadiran_db(nama):
    try:
        with app.app_context():
            now = datetime.now()
            tanggal, jam = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
            if Attendance.query.filter_by(nama=nama, tanggal=tanggal).first(): return None, None
            
            batas = app_settings.get('batas_waktu', '08:30:00')
            batas_waktu = datetime.strptime(batas, "%H:%M:%S").time()
            status = "Tepat Waktu" if now.time() <= batas_waktu else "Terlambat"
            
            new_log = Attendance(nama=nama, tanggal=tanggal, waktu=jam, status=status)
            db.session.add(new_log)
            db.session.commit()
            return jam, status
    except: return None, None

def camera_worker():
    global output_frame, wajah_sudah_absen, tanggal_terakhir_absen, latest_scan_event
    global registration_request, registration_response
    global total_scans, total_confidence, total_inference_time
    
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        print("PERINGATAN: Tidak ada kamera fisik terdeteksi (Mode Cloud Berjalan).")
    
    frame_counter = 0
    cached_face_locations, cached_colors, cached_display_texts, cached_metrics = [], [], [], []
    active_challenges = {}
    available_actions = ["KEDIPKAN MATA", "BUKA MULUT", "TOLEH KIRI", "TOLEH KANAN"]

    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.05); continue
            frame = cv2.flip(frame, 1)

            raw_frame = frame.copy()
            frame_counter += 1
            
            if registration_request is not None:
                name_to_register = registration_request
                try:
                    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb_frame, model="cnn")
                    if len(locs) == 0:
                        registration_response = {"status": "error", "message": "Wajah tidak terdeteksi!"}
                    else:
                        landmarks = face_recognition.face_landmarks(rgb_frame, locs)[0]
                        new_encoding = align_and_encode_face(rgb_frame, landmarks)
                        if new_encoding is None: new_encoding = face_recognition.face_encodings(rgb_frame, locs)[0]
                            
                        encoding_json = json.dumps(new_encoding.tolist())
                        filepath = os.path.join(DATASET_DIR, f"{name_to_register}.jpg")
                        cv2.imwrite(filepath, raw_frame)
                        
                        with app.app_context():
                            new_user = User(nama=name_to_register, encoding=encoding_json)
                            db.session.add(new_user)
                            db.session.commit()
                            
                        known_face_encodings.append(new_encoding)
                        known_face_names.append(name_to_register)
                        registration_response = {"status": "success"}
                except Exception as e: registration_response = {"status": "error", "message": str(e)}
                registration_request = None 
                continue 

            tgl_sekarang = datetime.now().date()
            if tgl_sekarang > tanggal_terakhir_absen:
                wajah_sudah_absen.clear()
                tanggal_terakhir_absen = tgl_sekarang

            if frame_counter % 2 == 0:
                start_time = time.time() # Mulai Pencatatan Waktu Inferensi
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
                if len(face_locations) > 0:
                    try: face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                    except: face_landmarks_list = [{}] * len(face_locations)
                else: face_landmarks_list = []

                inference_time_ms = round((time.time() - start_time) * 1000) # Kalkulasi Waktu Inferensi Akhir
                
                cached_face_locations, cached_colors, cached_display_texts, cached_metrics = face_locations, [], [], []

                for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
                    name, color, display_text = "Unknown", (0, 0, 255), "Wajah Tidak Dikenali"
                    metric_text = f"Memproses AI: {inference_time_ms} ms" # Status Default
                    confidence = 0.0
                    
                    orig_top, orig_right, orig_bottom, orig_left = top*4, right*4, bottom*4, left*4
                    if is_spoof_texture(raw_frame, orig_top, orig_right, orig_bottom, orig_left):
                        color, display_text = (0, 0, 255), "SPOOF TERDETEKSI (Tekstur!)"
                        if name in active_challenges: del active_challenges[name]
                    else:
                        face_encoding = align_and_encode_face(rgb_small_frame, face_landmarks)
                        if face_encoding is None: face_encoding = face_recognition.face_encodings(rgb_small_frame, [(top, right, bottom, left)])[0]

                        ear, mar, pose = 1.0, 0.0, "TENGAH"
                        if 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
                            ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0
                        if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
                            mar = mouth_aspect_ratio(face_landmarks['top_lip'], face_landmarks['bottom_lip'])
                        pose = detect_head_pose(face_landmarks)

                        if known_face_encodings:
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            if len(face_distances) > 0:
                                best_match_idx = np.argmin(face_distances)
                                dist = face_distances[best_match_idx]
                                
                                if dist < FACE_MATCH_TOLERANCE:
                                    name = known_face_names[best_match_idx]
                                    confidence = round((1.0 - dist) * 100, 1) # Kalkulasi Confidence Score (0-100%)
                                    metric_text = f"Kecocokan: {confidence}% | Inferensi: {inference_time_ms} ms"
                                    
                                    # Akumulasi Data untuk Dashboard Statistik
                                    total_scans += 1
                                    total_confidence += confidence
                                    total_inference_time += inference_time_ms
                                    
                                    if name in wajah_sudah_absen:
                                        color, display_text = (255, 255, 0), f"{name} (Selesai)"
                                        if name in active_challenges: del active_challenges[name]
                                    else:
                                        if name not in active_challenges:
                                            active_challenges[name] = {"sequence": random.sample(available_actions, 3), "current_step": 0, "sukses_frames": 0, "tutup_sebelumnya": False, "cooldown": 0 }
                                        
                                        ch = active_challenges[name]
                                        step = ch["current_step"]
                                        
                                        if ch["cooldown"] > 0:
                                            ch["cooldown"] -= 1
                                            color, display_text = (0, 255, 0), f"{name} | Lanjut..."
                                        else:
                                            tantangan = ch["sequence"][step]
                                            color, display_text = (0, 255, 255), f"{name} | {step + 1}/3: {tantangan}!"
                                            passed = False
                                            
                                            ear_thresh = float(app_settings.get('ear_threshold', 0.20))
                                            mar_thresh = float(app_settings.get('mar_threshold', 0.45))

                                            if tantangan == "KEDIPKAN MATA":
                                                if ear < ear_thresh: ch["sukses_frames"] += 1; ch["tutup_sebelumnya"] = ch["sukses_frames"] >= 2
                                                elif ch["tutup_sebelumnya"] and ear > ear_thresh + 0.02: passed = True; ch["sukses_frames"] = 0
                                            elif tantangan == "BUKA MULUT" and mar > mar_thresh:
                                                ch["sukses_frames"] += 1; passed = ch["sukses_frames"] >= 3
                                            elif tantangan == "TOLEH KANAN" and pose == "KANAN":
                                                ch["sukses_frames"] += 1; passed = ch["sukses_frames"] >= 2
                                            elif tantangan == "TOLEH KIRI" and pose == "KIRI":
                                                ch["sukses_frames"] += 1; passed = ch["sukses_frames"] >= 2

                                            if passed:
                                                ch["current_step"] += 1
                                                ch["sukses_frames"], ch["tutup_sebelumnya"], ch["cooldown"] = 0, False, 10
                                                ucapkan_pesan("Bagus")
                                                
                                                if ch["current_step"] >= len(ch["sequence"]):
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
                    cached_metrics.append(metric_text)

            for (top, right, bottom, left), color, display_text, metric in zip(cached_face_locations, cached_colors, cached_display_texts, cached_metrics):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 75), color, cv2.FILLED)
                cv2.putText(frame, display_text, (left + 10, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0) if color == (0, 255, 255) else (255, 255, 255), 2)
                cv2.putText(frame, metric, (left + 10, bottom + 60), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0) if color == (0, 255, 255) else (220, 220, 220), 1)

            with frame_lock: output_frame = frame.copy()
        except: time.sleep(0.1)

t = threading.Thread(target=camera_worker, daemon=True)
t.start()

def generate_web_stream():
    global output_frame
    while True:
        frame_copy = None
        with frame_lock:
            if output_frame is not None: frame_copy = output_frame.copy()
            
        if frame_copy is None: 
            # Jika tidak ada kamera, buat gambar hitam berisi teks informasi
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "MODE CLOUD SERVER", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(blank, "(Kamera hanya aktif di perangkat lokal)", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1) # Refresh lambat agar server cloud tidak berat
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        admin = Admin.query.filter_by(username=request.form.get('username')).first()
        if admin and check_password_hash(admin.password, request.form.get('password')):
            session['logged_in'] = True
            session['username'] = admin.username
            session['role'] = admin.role
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Kredensial salah!"})
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    logs = Attendance.query.order_by(Attendance.id.desc()).all()
    users = User.query.order_by(User.id.desc()).all()
    tgl_hari_ini = datetime.now().strftime("%Y-%m-%d")
    hadir = Attendance.query.filter_by(tanggal=tgl_hari_ini).count()
    tepat = Attendance.query.filter_by(tanggal=tgl_hari_ini, status='Tepat Waktu').count()
    telat = Attendance.query.filter_by(tanggal=tgl_hari_ini, status='Terlambat').count()
    data_render = [(l.id, l.nama, l.tanggal, l.waktu, l.status) for l in logs]
    
    employee_status = []
    for u in users:
        if u.nama in wajah_sudah_absen: employee_status.append({"nama": u.nama, "status": "Hadir", "waktu": wajah_sudah_absen[u.nama]})
        else: employee_status.append({"nama": u.nama, "status": "Belum Hadir", "waktu": "-"})
        
    users_render = [(u.nama, u.waktu_daftar) for u in users]
    
    # Kalkulasi Metrik Global untuk Dasbor
    avg_acc = f"{(total_confidence / total_scans):.1f}%" if total_scans > 0 else "0.0%"
    avg_time = f"{(total_inference_time / total_scans):.0f} ms" if total_scans > 0 else "0 ms"
    
    return render_template('dashboard.html', data=data_render, users=users_render, hadir=hadir, tepat=tepat, telat=telat, employees=employee_status, settings=app_settings, avg_acc=avg_acc, avg_time=avg_time)

@app.route('/api/chart_data')
@login_required
def chart_data():
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    hadir_per_day = [Attendance.query.filter_by(tanggal=d).count() for d in dates]
    tgl_hari_ini = datetime.now().strftime("%Y-%m-%d")
    tepat = Attendance.query.filter_by(tanggal=tgl_hari_ini, status='Tepat Waktu').count()
    telat = Attendance.query.filter_by(tanggal=tgl_hari_ini, status='Terlambat').count()
    total_users = User.query.count()
    belum_hadir = total_users - (tepat + telat)
    if belum_hadir < 0: belum_hadir = 0

    return jsonify({
        "bar_labels": dates,
        "bar_data": hadir_per_day,
        "pie_data": [tepat, telat, belum_hadir]
    })

@app.route('/update_settings', methods=['POST'])
@superadmin_required
def update_settings():
    try:
        keys = ['batas_waktu', 'ear_threshold', 'mar_threshold', 'spoof_threshold']
        for k in keys:
            val = request.form.get(k)
            if val:
                setting = Setting.query.filter_by(key=k).first()
                if setting: setting.value = val
        db.session.commit()
        load_settings() 
        return jsonify({"status": "success", "message": "Pengaturan berhasil diperbarui!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/export_csv')
@login_required
def export_csv():
    logs = Attendance.query.order_by(Attendance.id.desc()).all()
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Nama Entitas', 'Tanggal', 'Jam Masuk', 'Status Kehadiran'])
    for l in logs: cw.writerow([l.id, l.nama, l.tanggal, l.waktu, l.status])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=Laporan_Kehadiran_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/scanner')
def scanner(): return render_template('scanner.html')
@app.route('/video_feed')
def video_feed(): return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_attendance')
def stream_attendance():
    def event_stream():
        last_sent_time = 0
        while True:
            with frame_lock: current_event = latest_scan_event.copy() if latest_scan_event else None
            if current_event and current_event.get("timestamp", 0) > last_sent_time:
                last_sent_time = current_event["timestamp"]
                yield f"data: {json.dumps(current_event)}\n\n"
            time.sleep(0.5)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/dataset/<path:filename>')
def serve_dataset(filename): return send_from_directory(DATASET_DIR, filename)

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
@superadmin_required 
def register_face():
    global registration_request, registration_response
    name = request.form.get('name')
    if not name: return jsonify({"status": "error", "message": "Nama tidak boleh kosong!"})
    if User.query.filter_by(nama=name).first(): return jsonify({"status": "error", "message": f"Wajah atas nama '{name}' sudah ada!"})
        
    registration_response = None
    registration_request = name
    timeout = 150 
    while registration_response is None and timeout > 0:
        time.sleep(0.1); timeout -= 1
        
    if registration_response is None:
        registration_request = None 
        return jsonify({"status": "error", "message": "Kamera sibuk, gagal memproses!"})
    return jsonify(registration_response)

@app.route('/delete_face', methods=['POST'])
@superadmin_required 
def delete_face():
    global known_face_encodings, known_face_names
    name = request.form.get('name')
    filepath = os.path.join(DATASET_DIR, f"{name}.jpg")
    if os.path.exists(filepath): os.remove(filepath)
    user = User.query.filter_by(nama=name).first()
    if user:
        db.session.delete(user)
        db.session.commit()
    if name in known_face_names:
        idx = known_face_names.index(name)
        known_face_names.pop(idx)
        known_face_encodings.pop(idx)
    return jsonify({"status": "success", "message": f"Data '{name}' berhasil dihapus!"})

if __name__ == '__main__':
    app.run(debug=False, port=5000)