"""
label_manual.py — Alat bantu pelabelan manual untuk validasi Cohen's Kappa.

Menampilkan tweet satu per satu. Anda cukup menekan ENTER bila setuju dengan
usulan label, atau mengetik angka bila tidak. Progres tersimpan otomatis,
jadi boleh berhenti kapan saja dan dilanjutkan nanti.

    python label_manual.py                    # file gold terbaru
    python label_manual.py --gold data/gold/gold_sample_xxx.csv

CATATAN METODOLOGI
Usulan label berasal dari lexicon (pra-anotasi otomatis). Keputusan akhir
tetap di tangan Anda. Bila cara ini dipakai, tuliskan di bab metodologi
sebagai "pelabelan manual dengan bantuan pra-anotasi otomatis" — bukan
"pelabelan otomatis", dan bukan pula "pelabelan manual murni".

Jangan menekan ENTER terus-menerus tanpa membaca. Bila itu dilakukan, yang
tercatat hanyalah label lexicon, dan nilai Kappa menjadi tidak bermakna.
"""

import argparse
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
LABEL_MAP = {0: "negatif", 1: "netral", 2: "positif"}

PANDUAN = """
PANDUAN PELABELAN
=================

  0 = NEGATIF   keluhan, kekecewaan, kritik, masalah, kerugian
  1 = NETRAL    pertanyaan, informasi, berita, promosi, tanpa opini jelas
  2 = POSITIF   pujian, kepuasan, terima kasih, rekomendasi

Aturan yang membantu saat ragu:

  - Tidak ada opini sama sekali (berita/info/pertanyaan)     -> 1 netral
  - Menyebut Traveloka tapi bukan menilai Traveloka          -> 1 netral
  - Iklan/promo tanpa pengalaman pribadi                     -> 1 netral
  - Campuran positif dan negatif, seimbang                   -> 1 netral
  - Sarkasme ("mantap, uang saya hilang")                    -> ikuti MAKSUD
  - Ragu-ragu antara dua label                               -> pilih 1 netral

Nilai berdasarkan SIKAP TERHADAP TRAVELOKA, bukan suasana hati penulis.

Perintah: ENTER=setuju  0/1/2=ubah  s=lewati  u=ulangi sebelumnya  q=simpan & keluar
"""


def latest_gold():
    f = glob.glob(os.path.join(GOLD_DIR, "gold_sample_*.csv"))
    return max(f, key=os.path.getmtime) if f else None


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description="Pelabelan manual untuk validasi Kappa")
    ap.add_argument("--gold", default=None, help="File gold CSV")
    args = ap.parse_args()

    path = args.gold or latest_gold()
    if not path:
        sys.exit("ERROR: Tidak ada file gold.\n"
                 "Jalankan dulu: python main.py --sample-gold 200")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "label_manual" not in df.columns:
        df["label_manual"] = ""
    df["label_manual"] = df["label_manual"].astype("object")

    def belum(i):
        v = df.at[i, "label_manual"]
        return pd.isna(v) or str(v).strip() == ""

    print(PANDUAN)
    print(f"File   : {path}")
    total = len(df)
    sisa = sum(1 for i in df.index if belum(i))
    print(f"Sampel : {total}  |  belum dilabeli: {sisa}")
    print(f"Perkiraan waktu: sekitar {sisa * 12 // 60} menit\n")
    input("Tekan ENTER untuk mulai... ")

    urutan = [i for i in df.index if belum(i)]
    pos = 0
    diubah = 0

    while pos < len(urutan):
        i = urutan[pos]
        auto = int(df.at[i, "label_auto"])
        sudah = total - sum(1 for j in df.index if belum(j))

        print("\n" + "=" * 68)
        print(f"[{sudah + 1}/{total}]  sisa {len(urutan) - pos}")
        print("=" * 68)
        print(f"\n{str(df.at[i, 'text']).strip()}\n")
        print("-" * 68)
        print(f"usulan lexicon: {auto} ({LABEL_MAP[auto]})")

        jawab = input("label [ENTER=setuju / 0,1,2 / s / u / q] > ").strip().lower()

        if jawab == "q":
            break
        if jawab == "s":
            pos += 1
            continue
        if jawab == "u":
            pos = max(0, pos - 1)
            df.at[urutan[pos], "label_manual"] = ""
            continue
        if jawab == "":
            df.at[i, "label_manual"] = auto
        elif jawab in ("0", "1", "2"):
            df.at[i, "label_manual"] = int(jawab)
            if int(jawab) != auto:
                diubah += 1
                print(f"  -> dikoreksi menjadi {LABEL_MAP[int(jawab)]}")
        else:
            print("  input tidak dikenal, diulang")
            continue

        pos += 1
        if pos % 10 == 0:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  [progres disimpan]")

    df.to_csv(path, index=False, encoding="utf-8-sig")
    selesai = sum(1 for i in df.index if not belum(i))

    print("\n" + "=" * 68)
    print(f"Tersimpan : {path}")
    print(f"Selesai   : {selesai}/{total}")
    print(f"Dikoreksi : {diubah} label berbeda dari usulan lexicon")
    if selesai:
        print(f"Tingkat koreksi: {diubah / selesai * 100:.1f}%")
        if diubah == 0:
            print("\nPERHATIAN: tidak ada satu pun koreksi.")
            print("Bila ini karena menekan ENTER tanpa membaca, nilai Kappa")
            print("tidak akan bermakna. Periksa kembali sebelum melanjutkan.")
    if selesai < total:
        print(f"\nBelum selesai. Jalankan lagi untuk melanjutkan dari sisa {total - selesai}.")
    else:
        print("\nLangkah berikutnya:")
        print("  python main.py --kappa")


if __name__ == "__main__":
    main()
