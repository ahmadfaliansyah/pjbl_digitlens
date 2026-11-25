from pathlib import Path
from hashlib import sha256


# Lokasi utama proyek
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder penyimpanan gambar
UPLOAD_DIR = PROJECT_ROOT / "storage" / "uploads"


def save_uploaded_image(
    image_bytes: bytes,
    original_filename: str
) -> str:

    # Ambil ekstensi file
    extension = Path(original_filename).suffix.lower()

    if extension not in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Format gambar tidak didukung.")

    # Buat folder jika belum tersedia
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Nama file hanya berdasarkan hash isi gambar. Gambar yang sama akan
    # menghasilkan nama yang sama, sehingga tidak tersimpan berulang kali.
    image_hash = sha256(image_bytes).hexdigest()[:16]

    filename = f"{image_hash}{extension}"

    # Lokasi penyimpanan
    file_path = UPLOAD_DIR / filename

    # Hindari menulis ulang file yang sama
    if not file_path.exists():
        file_path.write_bytes(image_bytes)

    # Kembalikan path relatif
    return file_path.relative_to(PROJECT_ROOT).as_posix()