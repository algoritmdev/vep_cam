from db import init_db, save_sequence, load_sequence

def main():
    init_db()
    print("📌 Admin Sequence CLI")
    print("=====================")

    while True:
        print("\n1️⃣ Hozirgi tartibni ko‘rish")
        print("2️⃣ Yangi tartib kiritish")
        print("3️⃣ Chiqish")

        choice = input("Tanlang (1/2/3): ")

        if choice == "1":
            seq = load_sequence()
            if seq:
                print("📋 Hozirgi tartib:", " → ".join(seq))
            else:
                print("⚠️ Tartib hali kiritilmagan.")
        elif choice == "2":
            names = input("Odamlarni vergul bilan kiriting (masalan: ali, vali, guli): ")
            seq_list = [n.strip().lower() for n in names.split(",") if n.strip()]
            if seq_list:
                save_sequence(seq_list)
                print("✅ Yangi tartib saqlandi:", " → ".join(seq_list))
            else:
                print("⚠️ Bo‘sh ro‘yxat kiritildi.")
        elif choice == "3":
            print("🚪 Chiqildi.")
            break
        else:
            print("❌ Noto‘g‘ri tanlov.")

if __name__ == "__main__":
    main()
