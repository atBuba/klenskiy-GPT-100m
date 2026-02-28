"""
💾 Memory-Mapped Dataset — интегрирован с ruwiki_full.txt

При 8GB данных мы не можем загрузить всё в RAM.
Решение: memory-mapped файл (mmap) + потоковая токенизация.

ОБНОВЛЕНО:
- Путь к датасету: data/ruwiki_full.txt (скачивается через download_data.py)
- Путь к токенизатору: tokenizer/tokenizer.model (32K BPE)
- Потоковая токенизация: записываем чанки на диск, а не в RAM
- Корректная обработка <|endoftext|> как разделителя документов

💡 ЗАМЕТКА ТИМЛИДА:
Старая версия копила все токены в list (all_ids.extend) — на 8GB
это ~2B токенов × 8 байт (int64 в Python list) ≈ 16GB RAM. OOM!
Новая версия пишет чанки напрямую в файл через numpy — RAM не растёт.
"""

import os
import numpy as np
from pathlib import Path
from tokenizer import Tokenizer


# Пути к данным (внутри репозитория)
REPO_ROOT = Path(__file__).parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "ruwiki_full.txt"
DATA_DIR = REPO_ROOT / "data"


def prepare_dataset(
    corpus_file: str = None,
    tokenizer_path: str = None,
    val_fraction: float = 0.005,  # 0.5% на валидацию (при 8GB — это ~40MB)
):
    """
    Токенизирует корпус и сохраняет как .bin файлы.
    Использует ПОТОКОВУЮ запись — не загружает все токены в RAM.

    corpus_file: путь к текстовому файлу (default: ruwiki_full.txt)
    tokenizer_path: путь к .model файлу SentencePiece (default: tokenizer/tokenizer.model)
    val_fraction: доля валидации
    """
    if corpus_file is None:
        corpus_file = str(DEFAULT_CORPUS)
    if tokenizer_path is None:
        tokenizer_path = str(REPO_ROOT / "tokenizer" / "tokenizer.model")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_bin = DATA_DIR / "train.bin"
    val_bin = DATA_DIR / "val.bin"

    if train_bin.exists() and val_bin.exists():
        train_tokens = np.memmap(train_bin, dtype=np.uint16, mode='r')
        val_tokens = np.memmap(val_bin, dtype=np.uint16, mode='r')
        print(f"✅ Датасет уже подготовлен:")
        print(f"   Train: {len(train_tokens):,} токенов ({train_bin.stat().st_size / 1024**3:.2f} GB)")
        print(f"   Val:   {len(val_tokens):,} токенов ({val_bin.stat().st_size / 1024**3:.2f} GB)")
        return

    if not os.path.exists(corpus_file):
        print(f"❌ Файл корпуса не найден: {corpus_file}")
        print(f"   Ожидаемый путь: {DEFAULT_CORPUS}")
        return

    print(f"🔄 Подготовка датасета...")
    tokenizer = Tokenizer(tokenizer_path)
    print(f"   Токенизатор: {tokenizer.vocab_size} токенов")
    print(f"   Корпус: {corpus_file}")

    # ── Потоковая токенизация ──
    # Вместо all_ids.extend() (OOM на 8GB!) пишем чанки напрямую в temp файл
    chunk_size = 10 * 1024 * 1024  # 10MB текста за раз
    temp_bin = DATA_DIR / "all_tokens.bin"

    file_size = os.path.getsize(corpus_file)
    processed = 0
    total_tokens = 0

    print(f"   Токенизация {file_size / 1024**3:.2f} GB текста...")
    print(f"   (потоковая запись в {temp_bin})")

    with open(temp_bin, "wb") as out_f:
        with open(corpus_file, "r", encoding="utf-8") as in_f:
            buffer = ""
            while True:
                chunk = in_f.read(chunk_size)
                if not chunk:
                    # Токенизируем оставшийся буфер
                    if buffer:
                        ids = tokenizer.encode(buffer)
                        arr = np.array(ids, dtype=np.uint16)
                        arr.tofile(out_f)
                        total_tokens += len(ids)
                    break

                # Ищем последний <|endoftext|> в чанке для чистого разреза
                buffer += chunk
                last_sep = buffer.rfind("<|endoftext|>")
                if last_sep != -1:
                    # Токенизируем до разделителя включительно
                    to_process = buffer[:last_sep + len("<|endoftext|>")]
                    buffer = buffer[last_sep + len("<|endoftext|>"):]
                else:
                    # Нет разделителя — токенизируем весь буфер, оставляя хвост
                    # (чтобы не разрезать слова)
                    safe_cut = buffer.rfind(" ", 0, len(buffer) - 100)
                    if safe_cut == -1:
                        safe_cut = len(buffer)
                    to_process = buffer[:safe_cut]
                    buffer = buffer[safe_cut:]

                ids = tokenizer.encode(to_process)
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(out_f)
                total_tokens += len(ids)

                processed += len(chunk.encode('utf-8'))
                print(f"\r   {processed / 1024**3:.2f} / {file_size / 1024**3:.2f} GB "
                      f"({processed * 100 / file_size:.1f}%) "
                      f"| {total_tokens:,} токенов", end="", flush=True)

    print()
    print(f"   Всего: {total_tokens:,} токенов")
    print(f"   Compression ratio: {file_size / (total_tokens * 2):.1f}x (текст → uint16)")

    # ── Разделяем на train/val ──
    all_data = np.memmap(temp_bin, dtype=np.uint16, mode='r')
    val_size = int(len(all_data) * val_fraction)
    train_size = len(all_data) - val_size

    print(f"   Splitting: train={train_size:,} | val={val_size:,}")

    # Записываем train
    train_data = np.memmap(train_bin, dtype=np.uint16, mode='w+', shape=(train_size,))
    train_data[:] = all_data[:train_size]
    train_data.flush()

    # Записываем val
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='w+', shape=(val_size,))
    val_data[:] = all_data[train_size:]
    val_data.flush()

    # Удаляем временный файл
    temp_bin.unlink()

    print(f"\n✅ Датасет сохранён:")
    print(f"   Train: {train_size:,} токенов ({train_bin.stat().st_size / 1024**3:.2f} GB)")
    print(f"   Val:   {val_size:,} токенов ({val_bin.stat().st_size / 1024**3:.2f} GB)")


def prepare_test_dataset(
    corpus_file: str = None,
    tokenizer_path: str = None,
    target_tokens: int = 10_000_000,  # ~10M токенов
    val_fraction: float = 0.01,       # 1% на валидацию (больше % т.к. датасет маленький)
):
    """
    🧪 Тестовый режим: создаёт маленький датасет (~10M токенов) из начала корпуса.

    Для быстрой проверки что обучение работает:
    - Токенизируется только начало файла (не весь 8GB корпус)
    - ~10M токенов ≈ 20MB на диске (uint16)
    - Токенизация занимает ~10-30 секунд вместо 30-40 минут

    Использование:
        python dataset.py --test              # 10M токенов (по умолчанию)
        python dataset.py --test --tokens 5M  # 5M токенов
        python dataset.py --test --tokens 1M  # 1M токенов (совсем быстро)
    """
    if corpus_file is None:
        corpus_file = str(DEFAULT_CORPUS)
    if tokenizer_path is None:
        tokenizer_path = str(REPO_ROOT / "tokenizer" / "tokenizer.model")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_bin = DATA_DIR / "train.bin"
    val_bin = DATA_DIR / "val.bin"

    # В тестовом режиме ПЕРЕЗАПИСЫВАЕМ существующие файлы
    # (чтобы можно было переключаться между test и full)

    if not os.path.exists(corpus_file):
        print(f"❌ Файл корпуса не найден: {corpus_file}")
        print(f"   Ожидаемый путь: {DEFAULT_CORPUS}")
        return

    print(f"🧪 ТЕСТОВЫЙ РЕЖИМ — создаю мини-датасет (~{target_tokens/1e6:.0f}M токенов)")
    tokenizer = Tokenizer(tokenizer_path)
    print(f"   Токенизатор: {tokenizer.vocab_size} токенов")
    print(f"   Корпус: {corpus_file}")

    # Читаем текст порциями, пока не наберём нужное количество токенов
    # Для 10M токенов нужно ~20-40MB текста (зависит от языка)
    chunk_size = 5 * 1024 * 1024  # 5MB за раз
    temp_bin = DATA_DIR / "all_tokens.bin"
    total_tokens = 0

    print(f"   Цель: {target_tokens:,} токенов")

    with open(temp_bin, "wb") as out_f:
        with open(corpus_file, "r", encoding="utf-8") as in_f:
            buffer = ""
            while total_tokens < target_tokens:
                chunk = in_f.read(chunk_size)
                if not chunk:
                    # Файл закончился раньше цели
                    if buffer:
                        ids = tokenizer.encode(buffer)
                        arr = np.array(ids, dtype=np.uint16)
                        arr.tofile(out_f)
                        total_tokens += len(ids)
                    break

                buffer += chunk
                last_sep = buffer.rfind("<|endoftext|>")
                if last_sep != -1:
                    to_process = buffer[:last_sep + len("<|endoftext|>")]
                    buffer = buffer[last_sep + len("<|endoftext|>"):]
                else:
                    safe_cut = buffer.rfind(" ", 0, len(buffer) - 100)
                    if safe_cut == -1:
                        safe_cut = len(buffer)
                    to_process = buffer[:safe_cut]
                    buffer = buffer[safe_cut:]

                ids = tokenizer.encode(to_process)
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(out_f)
                total_tokens += len(ids)

                print(f"\r   {total_tokens:,} / {target_tokens:,} токенов "
                      f"({total_tokens * 100 / target_tokens:.0f}%)", end="", flush=True)

    print()
    print(f"   Итого: {total_tokens:,} токенов")

    # Обрезаем до target_tokens (могли набрать чуть больше)
    all_data = np.memmap(temp_bin, dtype=np.uint16, mode='r')
    actual_size = min(len(all_data), target_tokens)
    all_data = all_data[:actual_size]

    val_size = int(actual_size * val_fraction)
    train_size = actual_size - val_size

    print(f"   Splitting: train={train_size:,} | val={val_size:,}")

    # Записываем train
    train_data = np.memmap(train_bin, dtype=np.uint16, mode='w+', shape=(train_size,))
    train_data[:] = all_data[:train_size]
    train_data.flush()

    # Записываем val
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='w+', shape=(val_size,))
    val_data[:] = all_data[train_size:]
    val_data.flush()

    # Удаляем временный файл
    temp_bin.unlink()

    size_mb = (train_size + val_size) * 2 / 1024**2
    print(f"\n✅ Тестовый датасет сохранён ({size_mb:.1f} MB на диске):")
    print(f"   Train: {train_size:,} токенов ({train_bin.stat().st_size / 1024**2:.1f} MB)")
    print(f"   Val:   {val_size:,} токенов ({val_bin.stat().st_size / 1024**2:.1f} MB)")
    print(f"\n💡 Теперь запусти обучение:")
    print(f"   python train.py --test")


class MemmapDataset:
    """
    Memory-mapped датасет для обучения.

    Загружает .bin файл через numpy memmap — данные читаются с диска
    по мере необходимости, а не целиком в RAM.
    """

    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.length = len(self.data)

    def get_batch(self, batch_size: int, device):
        """Возвращает случайный батч (x, y)."""
        import torch

        max_start = self.length - self.block_size - 1
        starts = np.random.randint(0, max_start, size=(batch_size,))

        x = np.stack([self.data[s:s + self.block_size].astype(np.int64) for s in starts])
        y = np.stack([self.data[s + 1:s + self.block_size + 1].astype(np.int64) for s in starts])

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        return x, y


def _parse_token_count(s: str) -> int:
    """Парсит строку вида '10M', '1m', '500K', '500k' в число токенов."""
    s = s.strip().upper()
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--test" in args:
        # ── Тестовый режим ──
        args.remove("--test")

        # Парсим --tokens N
        target = 10_000_000  # по умолчанию 10M
        if "--tokens" in args:
            idx = args.index("--tokens")
            if idx + 1 < len(args):
                target = _parse_token_count(args[idx + 1])
                args.pop(idx + 1)
                args.pop(idx)

        corpus = args[0] if len(args) >= 1 else None
        tok = args[1] if len(args) >= 2 else None
        prepare_test_dataset(corpus, tok, target_tokens=target)

    else:
        corpus = args[0] if len(args) >= 1 else None
        tok = args[1] if len(args) >= 2 else None
        prepare_dataset(corpus, tok)
