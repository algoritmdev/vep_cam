import os
import sqlite3
import numpy as np
import cv2
import torch
import time
import threading
import queue
from facenet_pytorch import MTCNN, InceptionResnetV1
from playsound import playsound
import clr

# === PTZ DLL ===
try:
    clr.AddReference(r"C:\\Users\\b.qarahanov\\cam-master\\PTZDevice.dll")
    from PTZ import PTZDevice, PTZType
    device = PTZDevice.GetDevice("BCC950 ConferenceCam", PTZType.Relative)
except Exception as e:
    print(f"[⚠️] PTZ DLL ulanmagan: {e}")
    device = None

# === DB sozlamalari ===
DB_FILE = "people.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL,
        audio1 TEXT,
        audio2 TEXT,
        video TEXT
    )
    """)
    conn.commit()
    return conn, cursor

# === Yuz embedding olish ===
def get_embedding(img_path, mtcnn, resnet, device_torch):
    if not os.path.exists(img_path):
        print(f"[❌] Rasm topilmadi: {img_path}")
        return None

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)

    if face is None:
        print("[❌] Yuz topilmadi!")
        return None

    if face.ndim == 5:
        face = face.squeeze(0)
    if face.ndim == 3:
        face = face.unsqueeze(0)

    face = face.to(device_torch)

    with torch.no_grad():
        embedding = resnet(face).detach().cpu().numpy().astype(np.float32)

    return embedding.flatten()

# === Odam qo‘shish ===
def add_person(cursor, conn, name, image_path, audio1, audio2, video, resnet, mtcnn, device_torch):
    embedding = get_embedding(image_path, mtcnn, resnet, device_torch)
    if embedding is None:
        return

    cursor.execute(
        "INSERT INTO people (name, encoding, audio1, audio2, video) VALUES (?, ?, ?, ?, ?)",
        (name, embedding.tobytes(), audio1, audio2, video)
    )
    conn.commit()
    print(f"[✅] {name} bazaga qo‘shildi.")

# === DB’dan tanish yuzlarni olish ===
def get_known_faces(cursor):
    cursor.execute("SELECT id, name, encoding FROM people")
    rows = cursor.fetchall()
    encodings, names = [], []
    for row in rows:
        enc = np.frombuffer(row[2], dtype=np.float32).reshape(-1)
        encodings.append(enc)
        names.append(row[1])
    return encodings, names

# === Audio/Video xavfsiz ijro ===
def safe_play(path, label):
    if path and os.path.exists(path):
        try:
            print(f"[🔊] {label} ijro etilmoqda...")
            playsound(path)
        except Exception as e:
            print(f"[❌] {label} ijroda xato: {e}")
    else:
        print(f"[⚠️] {label} topilmadi: {path}")

def safe_play_video(path, label):
    if path and os.path.exists(path):
        print(f"[🎬] {label} ijro etilmoqda...")
        cap = cv2.VideoCapture(path)
        cv2.namedWindow("Media Player", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Media Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Media Player", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyWindow("Media Player")
    else:
        print(f"[⚠️] Video topilmadi: {path}")

# === Media Queue ===
media_queue = queue.Queue()
played_set = set()

def media_worker():
    while True:
        name, audio1, audio2, video = media_queue.get()
        if name in played_set:
            media_queue.task_done()
            continue

        print(f"[⏳] {name} uchun 30 soniya kutyapmiz...")
        time.sleep(5)

        safe_play(audio1, f"{name} - Audio1")
        time.sleep(3)

        safe_play(audio2, f"{name} - Audio2")
        time.sleep(3)

        safe_play_video(video, f"{name} - Video")

        played_set.add(name)
        media_queue.task_done()

threading.Thread(target=media_worker, daemon=True).start()

def start_media_with_delay(name, audio1, audio2, video):
    if name not in played_set:
        media_queue.put((name, audio1, audio2, video))

# === PTZ asinxron boshqaruv ===
def move_ptz_async(move_x, move_y):
    def worker():
        try:
            device.Move(int(move_x), int(move_y))
        except Exception as e:
            print(f"[⚠️] PTZ xato: {e}")
    threading.Thread(target=worker, daemon=True).start()

# === Kamera + PTZ ===
def start_camera(cursor, mtcnn, resnet, device_torch):
    screen_width, screen_height = 1080, 720
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

    known_encodings, known_names = get_known_faces(cursor)

    if not known_encodings:
        print("[⚠️] Bazada odam yo‘q. Avval odam qo‘shing!")
        return

    print("[📷] Kamera ishga tushdi. Q bosib chiqish mumkin.")

    last_time = time.time()
    process_interval = 0.05  # har 50ms (20 FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_time >= process_interval:
            last_time = now

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(img_rgb)

            if boxes is not None:
                faces = mtcnn(img_rgb)
                if faces is not None:
                    if faces.ndim == 5:
                        faces = faces.squeeze(1)

                    faces = faces.to(device_torch)
                    with torch.no_grad():
                        embeddings = resnet(faces).detach().cpu().numpy()

                    areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
                    max_idx = np.argmax(areas)
                    box = boxes[max_idx]
                    embedding = embeddings[max_idx]

                    x1, y1, x2, y2 = [int(v) for v in box]
                    name = "Unknown"

                    if known_encodings:
                        dists = np.linalg.norm(np.array(known_encodings) - embedding, axis=1)
                        min_idx = np.argmin(dists)
                        if dists[min_idx] < 0.9:
                            name = known_names[min_idx]
                            cursor.execute("SELECT audio1, audio2, video FROM people WHERE name=?", (name,))
                            row = cursor.fetchone()
                            if row:
                                audio1, audio2, video = row
                                start_media_with_delay(name, audio1, audio2, video)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, name, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                    # === PTZ Movement ===
                    if device:
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        x_off_center = x_center - (screen_width / 2)
                        y_off_center = y_center - (screen_height / 2)

                        move_x = np.clip(x_off_center / 200, -1, 1)
                        move_y = np.clip(-y_off_center / 150, -1, 1)

                        if abs(move_x) > 0.3 or abs(move_y) > 0.3:
                            move_ptz_async(move_x, move_y)

        cv2.imshow("PTZ FaceNet", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Konsol menyu ===
def main():
    conn, cursor = init_db()
    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device_torch)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device_torch)

    while True:
        print("\n=== Menu ===")
        print("1. Odam qo‘shish (rasm + audio + video)")
        print("2. Kamerani ishga tushirish (PTZ bilan)")
        print("3. Chiqish")
        choice = input("Tanlang (1/2/3): ")

        if choice == "1":
            name = input("Ism: ")
            path = input("Rasm yo‘li (masalan: images/bekzod.jpg): ")
            audio1 = input("Audio1 yo‘li (bo‘sh qoldirish mumkin): ") or None
            audio2 = input("Audio2 yo‘li (bo‘sh qoldirish mumkin): ") or None
            video = input("Video yo‘li (bo‘sh qoldirish mumkin): ") or None
            add_person(cursor, conn, name, path, audio1, audio2, video, resnet, mtcnn, device_torch)

        elif choice == "2":
            start_camera(cursor, mtcnn, resnet, device_torch)

        elif choice == "3":
            break
        else:
            print("❌ Noto‘g‘ri tanlov.")

    conn.close()

if __name__ == "__main__":
    main()
