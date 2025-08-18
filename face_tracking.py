import cv2
import torch
import time
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import pywinusb.hid as hid
from db import init_db, load_sequence

# ==== Init DB va sequence olish ====
init_db()
sequence = load_sequence()
if not sequence:
    sequence = ["ali", "vali", "guli"]  # Default
print("📌 Admin ketma-ketligi:", sequence)

# ==== FaceNet model ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==== Encodinglarni tayyorlash ====
known_encodings = {}
for name in sequence:
    path = f"encodings/{name}.jpg"
    if os.path.exists(path):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None:
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (160, 160))
            face_tensor = torch.tensor(face, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device) / 255.0
            embedding = resnet(face_tensor).detach().cpu().numpy()[0]
            known_encodings[name] = embedding
            print(f"✅ {name} encoding tayyorlandi")
        else:
            print(f"⚠️ {name}.jpg da yuz topilmadi")
    else:
        print(f"⚠️ {name}.jpg topilmadi (encodings/ ichiga joylang)")

def cosine_similarity(a, b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

# ==== Logitech BCC950 tekshirish (pywinusb orqali) ====
VENDOR_ID = 0x046D   # Logitech VID
PRODUCT_ID = 0x0838  # BCC950 PID (agar ishlamasa, sizga 0838 bo‘lishi mumkin)

ptz_mode = False
device_usb = None

devices = hid.find_all_hid_devices()
print("🔍 Qurilmalar ro‘yxati:")
for dev in devices:
    print(f"VID={dev.vendor_id:04X}, PID={dev.product_id:04X}, Name={dev.product_name}")
    if dev.vendor_id == VENDOR_ID and dev.product_id == PRODUCT_ID:
        device_usb = dev

if device_usb:
    device_usb.open()
    ptz_mode = True
    print("✅ Logitech BCC950 topildi → PTZ rejimi yoqildi")
else:
    print("⚠️ Logitech BCC950 topilmadi → Notebook kamerasi (simulyatsiya) ishlatilmoqda")

def send_visca(cmd):
    """USB orqali VISCA komandasi yuborish"""
    try:
        if device_usb:
            report = device_usb.find_output_reports()[0]
            report.set_raw_data([0x00] + cmd)
            report.send()
    except Exception as e:
        print("❌ USB xato:", e)

def move_camera(dx, dy, frame):
    """Kamerani odamga qarab burish"""
    threshold = 40
    if ptz_mode:
        if dx < -threshold: send_visca([0x81, 0x01, 0x06, 0x01, 0xFF])  # Chap
        elif dx > threshold: send_visca([0x81, 0x01, 0x06, 0x02, 0xFF])  # O‘ng
        if dy < -threshold: send_visca([0x81, 0x01, 0x06, 0x03, 0xFF])  # Yuqori
        elif dy > threshold: send_visca([0x81, 0x01, 0x06, 0x04, 0xFF])  # Past
    else:
        action = []
        if dx < -threshold: action.append("⬅️ CHAP")
        elif dx > threshold: action.append("➡️ O‘NG")
        if dy < -threshold: action.append("⬆️ YUQORI")
        elif dy > threshold: action.append("⬇️ PAST")
        text = " | ".join(action) if action else "✅ Markazda"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)

# ==== Kamera ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Kamera topilmadi!")

frame_center_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
frame_center_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

current_index = 0
last_switch_time = time.time()
WATCH_TIME = 10

print("🚀 Admin tartibida FaceNet PTZ kuzatuv boshlandi... (chiqish uchun 'q')")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    target_name = sequence[current_index]

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (160,160))
            face_tensor = torch.tensor(face, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device) / 255.0
            embedding = resnet(face_tensor).detach().cpu().numpy()[0]

            if target_name in known_encodings:
                sim = cosine_similarity(embedding, known_encodings[target_name])
                if sim > 0.65:
                    face_center_x = int((x1+x2)//2)
                    face_center_y = int((y1+y2)//2)
                    dx = face_center_x - frame_center_x
                    dy = face_center_y - frame_center_y
                    move_camera(dx, dy, frame)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame, target_name, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    if time.time() - last_switch_time > WATCH_TIME:
        current_index = (current_index + 1) % len(sequence)
        last_switch_time = time.time()
        print(f"➡️ Endi {sequence[current_index]} kuzatilmoqda")

    cv2.imshow("FaceNet PTZ Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ptz_mode:
    device_usb.close()
