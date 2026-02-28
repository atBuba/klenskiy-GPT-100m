"""
📥 Скачивание датасета для обучения klenskiy-GPT-1B

Датасет: Russian Wikipedia (полный текст, очищенный)
Источник: https://huggingface.co/datasets/atBuba/ruwiki-dataset

Два режима:
1. python download_data.py --tokenized   ← РЕКОМЕНДУЕТСЯ
   Скачивает готовые train.bin + val.bin (~2.3 GB)
   Можно сразу начинать обучение — токенизация не нужна!

2. python download_data.py --raw
   Скачивает сырой ruwiki_full.txt (~7.9 GB)
   После этого нужна токенизация: python dataset.py

3. python download_data.py --all
   Скачивает всё (и .txt, и .bin)

4. python download_data.py --check
   Проверяет что данные на месте
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
CORPUS_FILE = DATA_DIR / "ruwiki_full.txt"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

HF_DATASET_REPO = "atBuba/ruwiki-dataset"


def _get_hf():
    """Импортирует huggingface_hub, ставит если нет."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        return HfApi, hf_hub_download
    except ImportError:
        print("❌ Библиотека huggingface_hub не установлена!")
        print("   Установи: pip install huggingface_hub")
        sys.exit(1)


def check_dataset():
    """Проверяет наличие данных."""
    has_bins = TRAIN_BIN.exists() and VAL_BIN.exists()
    has_txt = CORPUS_FILE.exists()

    if has_bins:
        train_gb = TRAIN_BIN.stat().st_size / 1024**3
        val_mb = VAL_BIN.stat().st_size / 1024**2
        print(f"✅ Токенизированные данные найдены:")
        print(f"   train.bin: {train_gb:.2f} GB")
        print(f"   val.bin:   {val_mb:.1f} MB")
        print(f"\n   Готово к обучению: python train.py")

    if has_txt:
        txt_gb = CORPUS_FILE.stat().st_size / 1024**3
        print(f"✅ Сырой корпус найден: ruwiki_full.txt ({txt_gb:.2f} GB)")

    if not has_bins and not has_txt:
        print("❌ Данные не найдены!")
        print("   python download_data.py --tokenized   # скачать готовые .bin (рекомендуется)")
        print("   python download_data.py --raw         # скачать сырой .txt")
        return False

    return True


def _download_file(filename: str):
    """Скачивает один файл из HF репозитория в data/."""
    _, hf_hub_download = _get_hf()
    print(f"   ⬇️  {filename}...")
    downloaded = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
    )
    downloaded_path = Path(downloaded)
    target = DATA_DIR / filename

    # hf_hub_download может сохранить в подпапку — перемещаем
    if downloaded_path != target and downloaded_path.exists():
        downloaded_path.rename(target)

    size_mb = target.stat().st_size / 1024**2
    print(f"      ✅ {filename} ({size_mb:.1f} MB)")
    return target


def download_tokenized():
    """
    Скачивает готовые токенизированные файлы (train.bin + val.bin).
    После этого можно сразу запускать обучение — токенизация не нужна.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_BIN.exists() and VAL_BIN.exists():
        train_gb = TRAIN_BIN.stat().st_size / 1024**3
        print(f"✅ Токенизированные данные уже скачаны:")
        print(f"   train.bin: {train_gb:.2f} GB")
        print(f"\n   Готово к обучению: python train.py")
        return

    HfApi, _ = _get_hf()
    api = HfApi()

    # Проверяем что файлы есть в репозитории
    try:
        files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
    except Exception as e:
        print(f"❌ Ошибка при доступе к репозиторию: {e}")
        sys.exit(1)

    if "train.bin" not in files or "val.bin" not in files:
        print("⚠️  Токенизированные файлы (train.bin, val.bin) не найдены в репозитории!")
        print(f"   Доступные файлы: {[f for f in files if not f.startswith('.')]}")
        print(f"\n   Скачай сырой корпус и токенизируй:")
        print(f"   python download_data.py --raw")
        print(f"   python dataset.py")
        return

    print(f"📥 Скачиваю токенизированные данные с HuggingFace...")
    print(f"   Репозиторий: https://huggingface.co/datasets/{HF_DATASET_REPO}\n")

    _download_file("train.bin")
    _download_file("val.bin")

    train_gb = TRAIN_BIN.stat().st_size / 1024**3
    print(f"\n✅ Готово! Скачано {train_gb:.2f} GB токенизированных данных.")
    print(f"   Токенизация НЕ НУЖНА — можно сразу обучать:")
    print(f"   python train.py --test    # тестовый прогон")
    print(f"   python train.py           # полное обучение")


def download_raw():
    """Скачивает сырой текстовый корпус (ruwiki_full.txt)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_FILE.exists():
        size_gb = CORPUS_FILE.stat().st_size / 1024**3
        print(f"✅ Корпус уже скачан: ruwiki_full.txt ({size_gb:.2f} GB)")
        return

    HfApi, _ = _get_hf()
    api = HfApi()

    try:
        files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
    except Exception as e:
        print(f"❌ Ошибка при доступе к репозиторию: {e}")
        sys.exit(1)

    print(f"📥 Скачиваю сырой корпус с HuggingFace...")
    print(f"   Репозиторий: https://huggingface.co/datasets/{HF_DATASET_REPO}\n")

    # Ищем .txt файл
    txt_files = [f for f in files if f.endswith(".txt") and f != "README.md"]

    if txt_files:
        target_file = txt_files[0]
        _download_file(target_file)

        # Переименовываем если имя отличается
        downloaded = DATA_DIR / target_file
        if downloaded.exists() and downloaded != CORPUS_FILE:
            downloaded.rename(CORPUS_FILE)
    else:
        print(f"⚠️  .txt файл не найден в репозитории!")
        print(f"   Доступные файлы: {[f for f in files if not f.startswith('.')]}")
        return

    if CORPUS_FILE.exists():
        size_gb = CORPUS_FILE.stat().st_size / 1024**3
        print(f"\n✅ Корпус скачан: ruwiki_full.txt ({size_gb:.2f} GB)")
        print(f"\n   Следующий шаг — токенизация:")
        print(f"   python dataset.py            # полный датасет")
        print(f"   python dataset.py --test     # тестовый (~10M токенов)")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--check" in args:
        check_dataset()
    elif "--tokenized" in args:
        download_tokenized()
    elif "--raw" in args:
        download_raw()
    elif "--all" in args:
        download_tokenized()
        print()
        download_raw()
    else:
        print("=" * 60)
        print("📥 Скачивание датасета для klenskiy-GPT-1B")
        print(f"   Источник: https://huggingface.co/datasets/{HF_DATASET_REPO}")
        print("=" * 60)
        print()
        print("Использование:")
        print("  python download_data.py --tokenized   # готовые .bin (рекомендуется)")
        print("  python download_data.py --raw         # сырой .txt (~8 GB)")
        print("  python download_data.py --all         # всё")
        print("  python download_data.py --check       # проверить данные")
        print()
        print("Рекомендуемый путь (без токенизации):")
        print("  python download_data.py --tokenized")
        print("  python train.py --test")
