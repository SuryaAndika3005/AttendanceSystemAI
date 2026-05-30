import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import csv
import threading
import math
import ctypes

# ==========================================
# 1. KELAS MULTI-THREADING UNTUK KAMERA
# ==========================================
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ==========================================
# 2. FUNGSI LIVENESS DETECTION
# ==========================================
def hitung_jarak(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def eye_aspect_ratio(eye):
    A = hitung_jarak(eye[1], eye[5])
    B = hitung_jarak(eye[2], eye[4])
    C = hitung_jarak(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.22
mata_tertutup_sebelumnya = False

# ==========================================
# 3. KONFIGURASI DAN MEMORI SISTEM
# ==========================================
DATASET_DIR = "dataset_wajah"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

known_face_encodings = []
known_face_names = []

# PERUBAHAN: Sekarang menggunakan Dictionary untuk menyimpan Nama dan Jam Absen
wajah_sudah_absen = {} 
tanggal_terakhir_absen = datetime.now().date()

root = tk.Tk()
root.withdraw()

def load_registered_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    print("Memuat data wajah...")
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(DATASET_DIR, filename)
            name = os.path.splitext(filename)[0]
            
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

# FITUR BARU: Membaca CSV saat program pertama kali dibuka
def muat_absensi_hari_ini():
    global wajah_sudah_absen
    wajah_sudah_absen = {}
    file_csv = 'Laporan_Presensi.csv'
    
    if os.path.isfile(file_csv):
        tanggal_sekarang = datetime.now().strftime("%Y-%m-%d")
        with open(file_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # Melewati baris header
            for row in reader:
                if len(row) >= 3:
                    nama_csv, tgl_csv, jam_csv = row[0], row[1], row[2]
                    if tgl_csv == tanggal_sekarang:
                        wajah_sudah_absen[nama_csv] = jam_csv
        print(f"Memori dipulihkan: {len(wajah_sudah_absen)} orang sudah absen hari ini.")

load_registered_faces()
muat_absensi_hari_ini() # Panggil fungsi pemulihan memori

# PERUBAHAN: Fungsi ini sekarang mengembalikan 'jam' agar bisa disimpan di memori
def catat_kehadiran(nama):
    file_csv = 'Laporan_Presensi.csv'
    file_exists = os.path.isfile(file_csv)
    
    waktu_sekarang = datetime.now()
    tanggal = waktu_sekarang.strftime("%Y-%m-%d")
    jam = waktu_sekarang.strftime("%H:%M:%S")
    
    with open(file_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Nama', 'Tanggal', 'Waktu'])
        writer.writerow([nama, tanggal, jam])
    
    return jam

def register_new_face(frame):
    new_name = simpledialog.askstring("Registrasi", "Masukkan Nama Anda:")
    if new_name:
        file_path = os.path.join(DATASET_DIR, f"{new_name}.jpg")
        cv2.imwrite(file_path, frame)
        load_registered_faces()
        messagebox.showinfo("Sukses", f"Wajah {new_name} berhasil didaftarkan!")

# ==========================================
# 4. LOOP UTAMA PROGRAM
# ==========================================
print("Kamera aktif. Sistem presensi berjalan...")
vs = WebcamStream(src=0).start()

while True:
    frame = vs.read()
    if frame is None:
        continue

    # Reset harian otomatis
    tanggal_sekarang = datetime.now().date()
    if tanggal_sekarang > tanggal_terakhir_absen:
        wajah_sudah_absen.clear()
        tanggal_terakhir_absen = tanggal_sekarang
        print("Ganti hari. Data memori absensi di-reset.")

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        top *= 4; right *= 4; bottom *= 4; left *= 4

        name = "Unknown"
        color = (0, 0, 255)
        display_text = "Tidak Dikenal (Tekan 'R')"

        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < 0.45:
                name = known_face_names[best_match_index]
                
                # Cek apakah nama ada di dalam memori presensi hari ini
                if name in wajah_sudah_absen:
                    color = (255, 255, 0)
                    display_text = f"{name} (Absen pkl {wajah_sudah_absen[name]})"
                else:
                    color = (0, 255, 0)
                    display_text = f"Wajah: {name}"

                if ear < EAR_THRESHOLD:
                    mata_tertutup_sebelumnya = True
                elif ear >= EAR_THRESHOLD and mata_tertutup_sebelumnya:
                    mata_tertutup_sebelumnya = False
                    
                    if name not in wajah_sudah_absen:
                        # Jika belum absen, catat dan munculkan popup sukses SEKALI SAJA
                        jam_masuk = catat_kehadiran(name)
                        wajah_sudah_absen[name] = jam_masuk
                        threading.Thread(target=lambda n=name: ctypes.windll.user32.MessageBoxW(0, f"Kehadiran '{n}' tercatat!", "Presensi Sukses", 64), daemon=True).start()
                if not mata_tertutup_sebelumnya and name not in wajah_sudah_absen:
                    cv2.putText(frame, "Silakan Berkedip!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom + 35), color, cv2.FILLED)
        cv2.putText(frame, display_text, (left + 5, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if name == "Unknown":
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or key == ord('R'):
                register_new_face(frame)

    cv2.imshow('Sistem Presensi AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()