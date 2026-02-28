"""
📝 BPE-токенизатор (SentencePiece) — интегрирован с обученным токенизатором

Использует предобученный токенизатор из tokenizer/tokenizer.model
Обучен на ruwiki_full.txt (7.9 GB русской Wikipedia)

Конфигурация обученного токенизатора:
- Vocab: 32,000 BPE токенов
- Special tokens: <unk>=0, <s>=1 (BOS), </s>=2 (EOS), <|padding|>=3, <|endoftext|>=4, <|sep|>=5, <|mask|>=6
- Character coverage: 99.95% (Cyrillic + Latin)
- Byte fallback: True
- Split digits: True

💡 ЗАМЕТКА ТИМЛИДА:
Токенизатор уже обучен — не нужно обучать заново.
Мы просто загружаем .model файл и используем его.
"""

import os
import sentencepiece as spm
from pathlib import Path


# Путь к обученному токенизатору (внутри репозитория)
REPO_ROOT = Path(__file__).parent
TOKENIZER_MODEL_PATH = REPO_ROOT / "tokenizer" / "tokenizer.model"


class Tokenizer:
    """
    Обёртка над SentencePiece для удобства.

    Загружает предобученный токенизатор и предоставляет единый интерфейс
    для encode/decode с поддержкой специальных токенов.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(TOKENIZER_MODEL_PATH)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Токенизатор не найден: {model_path}\n"
                f"Убедись что tokenizer/tokenizer.model существует"
            )

        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # Базовые ID (из конфига обученного токенизатора)
        self.vocab_size = self.sp.get_piece_size()  # 32000
        self.unk_id = self.sp.piece_to_id("<unk>")  # 0
        self.bos_id = self.sp.piece_to_id("<s>")    # 1
        self.eos_id = self.sp.piece_to_id("</s>")   # 2
        self.pad_id = self.sp.piece_to_id("<|padding|>")    # 3
        self.eot_id = self.sp.piece_to_id("<|endoftext|>")  # 4 — разделитель документов
        self.sep_id = self.sp.piece_to_id("<|sep|>")        # 5
        self.mask_id = self.sp.piece_to_id("<|mask|>")       # 6

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Текст → список токен-ID."""
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Список токен-ID → текст."""
        return self.sp.decode(ids)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (f"Tokenizer(vocab_size={self.vocab_size}, "
                f"model='{os.path.basename(self.model_path)}')")


if __name__ == "__main__":
    # Быстрый тест токенизатора
    tok = Tokenizer()
    print(f"📝 {tok}")
    print(f"   BOS={tok.bos_id}, EOS={tok.eos_id}, PAD={tok.pad_id}, EOT={tok.eot_id}")
    print()

    examples = [
        "Все счастливые семьи похожи друг на друга",
        "Привет! Как дела?",
        "Трансформер — это архитектура нейросети",
        "GPT-4 был выпущен в 2023 году.",
    ]
    for text in examples:
        ids = tok.encode(text)
        pieces = tok.sp.encode(text, out_type=str)
        decoded = tok.decode(ids)
        print(f'   "{text}"')
        print(f"   → {pieces}")
        print(f"   → IDs: {ids[:10]}{'...' if len(ids) > 10 else ''} ({len(ids)} токенов)")
        print(f"   → Decoded: {decoded}")
        print()
