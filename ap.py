import pywinusb.hid as hid

# ==== Logitech BCC950 ID ====
VENDOR_ID = 0x046D  # Logitech VID
PRODUCT_ID = 0x0837  # PTZ boshqaruv Interface 0 uchun PID

ptz_mode = False
device_usb = None

devices = hid.find_all_hid_devices()
print("🔍 Qurilmalar ro‘yxati:")

for dev in devices:
    print(f"VID={dev.vendor_id:04X}, PID={dev.product_id:04X}, Path={dev.device_path}, Name={dev.product_name}")

    # Faqat Interface 0 ni tanlash
    if dev.vendor_id == VENDOR_ID and dev.product_id == PRODUCT_ID and "mi_00" in dev.device_path.lower():
        device_usb = dev

if device_usb:
    try:
        device_usb.open()
        ptz_mode = True
        print("✅ Logitech BCC950 (Interface 0) topildi → PTZ rejimi yoqildi")
    except Exception as e:
        print("❌ PTZ interfeysini ochishda xato:", e)
else:
    print("⚠️ PTZ uchun Interface 0 topilmadi → Notebook kamerasi (simulyatsiya) ishlatilmoqda")
