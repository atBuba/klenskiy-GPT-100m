"""
📤 Загрузка токенизированных данных на HuggingFace

Загружает train.bin и val.bin в репозиторий atBuba/ruwiki-dataset,
чтобы при обучении не тратить время на токенизацию (~30-40 минут).

Использование:
    # 1. Залогинься в HuggingFace (один раз):
    huggingface-cli login

    # 2. Запусти загрузку:
    python upload_to_hf.py /path/to/train.bin /path/to/val.bin

    # Или из папки с файлами:
    python upload_to_hf.py
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_DATASET_REPO = "atBuba/ruwiki-dataset"


def upload(train_bin: str, val_bin: str):
    api = HfApi()

    # Проверяем авторизацию
    try:
        user = api.whoami()["name"]
        print(f"✅ Авторизован как: {user}")
    except Exception:
        print("❌ Не залогинен в HuggingFace!")
        print("   Выполни: huggingface-cli login")
        sys.exit(1)

    train_path = Path(train_bin)
    val_path = Path(val_bin)

    if not train_path.exists():
        print(f"❌ Файл не найден: {train_path}")
        sys.exit(1)
    if not val_path.exists():
        print(f"❌ Файл не найден: {val_path}")
        sys.exit(1)

    train_size = train_path.stat().st_size / 1024**3
    val_size = val_path.stat().st_size / 1024**2
    print(f"\n📦 Файлы для загрузки:")
    print(f"   train.bin: {train_size:.2f} GB")
    print(f"   val.bin:   {val_size:.1f} MB")
    print(f"   Репозиторий: https://huggingface.co/datasets/{HF_DATASET_REPO}")

    # Загружаем train.bin
    print(f"\n⬆️  Загружаю train.bin ({train_size:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo="train.bin",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print("   ✅ train.bin загружен")

    # Загружаем val.bin
    print(f"⬆️  Загружаю val.bin ({val_size:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(val_path),
        path_in_repo="val.bin",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print("   ✅ val.bin загружен")

    print(f"\n🎉 Готово! Файлы доступны на:")
    print(f"   https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        upload(sys.argv[1], sys.argv[2])
    else:
        # По умолчанию ищем в текущей папке data/
        data_dir = Path(__file__).parent / "data"
        t = data_dir / "train.bin"
        v = data_dir / "val.bin"
        if t.exists() and v.exists():
            upload(str(t), str(v))
        else:
            print("Использование:")
            print("  python upload_to_hf.py /path/to/train.bin /path/to/val.bin")
            print("  python upload_to_hf.py   # если файлы в data/")
