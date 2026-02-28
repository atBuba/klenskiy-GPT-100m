"""
🏋️ Тренировочный цикл для 100M модели

Что изменилось по сравнению с учебной версией:

1. Mixed Precision (bf16) — в 2x быстрее, в 2x меньше памяти
2. Gradient Accumulation — имитация большого батча без доп. памяти
3. Cosine LR Schedule с warmup — стабильное обучение
4. Gradient Clipping — защита от "взрыва" градиентов
5. Memory-mapped Dataset — данные не грузятся в RAM
6. Checkpointing — сохраняем модель периодически
7. Wandb logging (опционально) — мониторинг обучения

💡 ЗАМЕТКА ТИМЛИДА:
Все эти техники — СТАНДАРТ в индустрии. Без них обучить 1B+ модель
практически невозможно. В реальности ещё добавляют:
- Distributed training (несколько GPU)
- FSDP / DeepSpeed (распределение модели между GPU)
- Flash Attention (оптимизированный kernel)
Но для одной GPU наш пайплайн достаточен.
"""

import os
import sys
import time
import math
import torch
from pathlib import Path
from contextlib import nullcontext

from model import MiniGPT, ModelConfig, TE_AVAILABLE
from dataset import MemmapDataset
from tokenizer import Tokenizer

# Transformer Engine (для FP8/FP4 на Blackwell/Hopper)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe

# ============================================================
# CUDA ОПТИМИЗАЦИИ (Blackwell / Hopper / Ampere)
# ============================================================
# TF32: использует Tensor Cores для float32 операций (~2x быстрее, точность ~FP16)
# На Blackwell RTX 5090 это БЕСПЛАТНОЕ ускорение для всех float32 вычислений.
# cudnn.benchmark: автоматически выбирает лучший CUDA алгоритм для convolutions.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 для матричных умножений
    torch.backends.cudnn.allow_tf32 = True          # TF32 для cuDNN операций
    torch.backends.cudnn.benchmark = True            # Авто-тюнинг cuDNN ядер
    torch.set_float32_matmul_precision('high')       # Предпочитать TF32 везде

# ============================================================
# КОНФИГУРАЦИЯ ОБУЧЕНИЯ
# ============================================================

# --- Данные ---
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
TOKENIZER_PATH = REPO_ROOT / "tokenizer" / "tokenizer.model"

# --- Обучение ---
MAX_STEPS = 50_000         # 100M модель обучается быстрее — 50K шагов достаточно
BATCH_SIZE = 16            # 100M модель ~10x меньше → batch можно увеличить
GRADIENT_ACCUM_STEPS = 8   # Кол-во микро-батчей перед обновлением весов
                           # Эффективный батч = BATCH_SIZE × GRADIENT_ACCUM_STEPS × block_size
                           # = 16 × 8 × 2048 = 262,144 токенов ≈ 0.26M

# --- Learning Rate ---
LEARNING_RATE = 3e-4       # Пиковый LR (выше для маленьких моделей — стандарт Chinchilla)
MIN_LR = 3e-5             # Минимальный LR (10% от пикового)
WARMUP_STEPS = 1000        # Разогрев (линейно растём от 0 до peak LR)
LR_DECAY_STEPS = 50_000    # Полный цикл cosine decay

# --- Cosine Annealing с Warm Restarts ---
# Вместо одного cosine decay до конца, делаем несколько циклов.
# После каждого цикла LR "перезапускается" — это помогает модели
# выбраться из локальных минимумов и найти лучшее решение.
# T_0 = длина первого цикла, T_mult = множитель (каждый цикл длиннее)
USE_WARM_RESTARTS = True   # True = warm restarts, False = стандартный cosine
RESTART_T0 = 10_000        # Длина первого цикла (шаги)
RESTART_T_MULT = 2         # Каждый следующий цикл в T_MULT раз длиннее
                           # Циклы: 10K → 20K → 40K (итого: 70K шагов покрывают 50K)

# --- Регуляризация ---
WEIGHT_DECAY = 0.1         # L2 регуляризация (стандарт для LLM)
GRAD_CLIP = 1.0            # Максимальная норма градиента

# --- Логирование ---
EVAL_INTERVAL = 500        # Как часто считаем val loss
EVAL_SAMPLES = 50          # Батчей для оценки
SAVE_INTERVAL = 5000       # Как часто сохраняем чекпоинт
SAMPLE_INTERVAL = 2000     # Как часто генерируем пример текста
LOG_INTERVAL = 50          # Как часто логируем train loss

# --- Чекпоинты ---
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# --- Wandb (опционально) ---
USE_WANDB = False
WANDB_PROJECT = "klenskiy-gpt-100m"

# --- Precision Mode (Blackwell optimization) ---
# "auto"  — автоматически определяет лучший режим по GPU (рекомендуется!)
# "bf16"  — стандарт (работает везде: Ampere, Hopper, Blackwell, MPS)
# "fp8"   — MXFP8 через Transformer Engine (Hopper H100+, Blackwell) — ~2x ускорение
# "fp4"   — NVFP4 через Transformer Engine (ТОЛЬКО серверные Blackwell B200/B100)
#
# ⚠️  FP4 (NVFP4BlockScaling) НЕ работает на consumer Blackwell (RTX 5090) —
#     Hadamard transform CUDA ядро несовместимо. Используй только на B200/B100.
#
# Для RTX 5090 (Blackwell consumer): "fp8" (auto определит)
# Для B200/B100 (Blackwell datacenter): "fp4" (задать вручную!)
# Для H100 (Hopper): "fp8"
# Для RTX 4090 (Ada): "bf16"
PRECISION_MODE = "auto"


def detect_precision_mode():
    """
    Автоматически определяет лучший precision mode по GPU.

    Blackwell consumer (SM 12.0, RTX 5090): FP8 (FP4 NVFP4 не работает на consumer GPU)
    Hopper (SM 9.0, H100): FP8
    Ada Lovelace (SM 8.9, RTX 4090): bf16
    Ampere (SM 8.0-8.6, A100/RTX 3090): bf16
    Остальные: bf16

    Для серверных Blackwell (B200/B100) задай PRECISION_MODE = "fp4" вручную.
    """
    if not torch.cuda.is_available() or not TE_AVAILABLE:
        return "bf16"

    # Compute capability: major.minor (например, 12.0 для Blackwell, 9.0 для Hopper)
    major, minor = torch.cuda.get_device_capability()

    if major >= 12:
        # Blackwell (RTX 5090 = SM 12.0, B200 = SM 12.0)
        # FP4 (NVFP4BlockScaling) вызывает ошибку на consumer RTX 5090:
        #   "CUDA Error: invalid argument" в hadamard_transform_cast_fusion.cu
        # Поэтому auto всегда выбирает FP8 для Blackwell.
        # Для серверных B200/B100 можно задать PRECISION_MODE = "fp4" вручную.
        print(f"🔍 GPU SM {major}.{minor} (Blackwell) → режим FP8")
        print(f"   (FP4 доступен только на серверных B200/B100, для RTX 5090 используем FP8)")
        return "fp8"
    elif major >= 9:
        # Hopper (H100 = SM 9.0) — FP8 нативно поддерживается
        print(f"🔍 GPU SM {major}.{minor} (Hopper) → режим FP8")
        return "fp8"
    else:
        # Ampere (SM 8.0), Ada Lovelace (SM 8.9), и старше
        print(f"🔍 GPU SM {major}.{minor} → режим bf16")
        return "bf16"

# --- Тестовый режим (--test) ---
# Переопределяется ниже если запустить: python train.py --test
TEST_MODE = False


def apply_test_config():
    """
    🧪 Тестовый режим: уменьшенные параметры для быстрой проверки.

    Цель — убедиться что весь пайплайн работает за 5-15 минут:
    - Меньше шагов (500 вместо 100K)
    - Меньше gradient accumulation (4 вместо 16)
    - Быстрый warmup (50 шагов)
    - Частое логирование и eval
    - bf16 вместо fp4 (не нужен Blackwell для теста)

    Запуск:
        python dataset.py --test           # создать тестовый датасет
        python train.py --test             # обучить на нём
    """
    global MAX_STEPS, BATCH_SIZE, GRADIENT_ACCUM_STEPS
    global LEARNING_RATE, MIN_LR, WARMUP_STEPS, LR_DECAY_STEPS
    global USE_WARM_RESTARTS, RESTART_T0, RESTART_T_MULT
    global EVAL_INTERVAL, EVAL_SAMPLES, SAVE_INTERVAL, SAMPLE_INTERVAL, LOG_INTERVAL
    global PRECISION_MODE, TEST_MODE

    TEST_MODE = True

    # Обучение
    MAX_STEPS = 500            # 500 шагов — достаточно увидеть падение loss
    BATCH_SIZE = 8             # 100M модель маленькая — batch 8 точно влезет
    GRADIENT_ACCUM_STEPS = 4   # Effective batch = 8 × 4 × 2048 = 65K токенов
    # При 10M токенов и 32K за шаг: 500 шагов = 16M токенов ≈ 1.6 эпохи

    # LR
    LEARNING_RATE = 3e-4       # Чуть выше — быстрее видно эффект на маленьком датасете
    MIN_LR = 3e-5
    WARMUP_STEPS = 50          # Быстрый warmup
    LR_DECAY_STEPS = 500

    # Без warm restarts (слишком мало шагов)
    USE_WARM_RESTARTS = False

    # Частое логирование
    EVAL_INTERVAL = 50         # Eval каждые 50 шагов
    EVAL_SAMPLES = 10          # Меньше батчей для eval (быстрее)
    SAVE_INTERVAL = 250        # Один чекпоинт в середине
    SAMPLE_INTERVAL = 100      # Генерация каждые 100 шагов
    LOG_INTERVAL = 10          # Логируем каждые 10 шагов

    # bf16 для теста (fp4/fp8 требуют спец. железо и могут быть нестабильны при коротких прогонах)
    PRECISION_MODE = "bf16"

    print("🧪 Тестовый режим: precision принудительно bf16")


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_lr(step):
    """
    Learning Rate Schedule с двумя режимами:

    1. Стандартный cosine decay (USE_WARM_RESTARTS=False):
       Warmup → плавный cosine decay до MIN_LR

    2. Cosine Annealing с Warm Restarts (USE_WARM_RESTARTS=True):
       Warmup → циклический cosine decay с перезапусками.
       Каждый цикл: LR падает от пикового до MIN_LR, затем "прыгает" обратно.
       Каждый следующий цикл длиннее в T_MULT раз.

       Зачем? Перезапуски помогают модели:
       - Выбираться из плохих локальных минимумов (LR jump выталкивает)
       - Исследовать разные области loss landscape
       - В финале — длинный цикл для тонкой настройки

       Статья: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent
       with Warm Restarts" (2017)
    """
    # Warmup (одинаковый для обоих режимов)
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    effective_step = step - WARMUP_STEPS

    if USE_WARM_RESTARTS:
        # Находим текущий цикл и позицию внутри него
        t_cur = effective_step
        t_i = RESTART_T0  # длина текущего цикла

        # Проходим по циклам: T0, T0*T_mult, T0*T_mult², ...
        while t_cur >= t_i:
            t_cur -= t_i
            t_i = int(t_i * RESTART_T_MULT)

        # Cosine decay внутри текущего цикла
        progress = t_cur / t_i
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)
    else:
        # Стандартный cosine decay
        total_decay_steps = LR_DECAY_STEPS - WARMUP_STEPS
        if effective_step >= total_decay_steps:
            return MIN_LR

        progress = effective_step / total_decay_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================

def setup():
    """
    Настраивает устройство и precision.

    Возвращает: (device, dtype, amp_ctx, te_ctx)
    - amp_ctx: torch.amp.autocast контекст (bf16/fp16 базовый)
    - te_ctx: transformer_engine.fp8_autocast контекст (для FP8/FP4)
              или nullcontext для bf16 режима
    """

    # Устройство
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {device_name} ({vram:.0f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 CPU (будет очень медленно для 1B!)")

    # ── Базовый AMP контекст (bf16 — всегда включён на CUDA) ──
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("⚡ Base precision: bf16")
    elif device.type == "cuda":
        dtype = torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        print("⚡ Base precision: fp16")
    else:
        dtype = torch.float32
        amp_ctx = nullcontext()
        print("📐 Full precision: fp32")

    # ── Transformer Engine контекст (FP8/FP4 поверх bf16) ──
    te_ctx_fn = None  # функция, создающая контекст (вызываем каждый шаг)

    # Авто-определение precision mode если задан "auto"
    global PRECISION_MODE
    if PRECISION_MODE == "auto":
        PRECISION_MODE = detect_precision_mode()

    if PRECISION_MODE in ("fp8", "fp4") and TE_AVAILABLE and device.type == "cuda":
        if PRECISION_MODE == "fp8":
            # MXFP8: DelayedScaling с FP8 E4M3/E5M2
            # Работает на Hopper (H100) и Blackwell (B200)
            fp8_recipe = te_recipe.DelayedScaling(
                margin=0,
                fp8_format=te_recipe.Format.HYBRID,  # E4M3 forward, E5M2 backward
                amax_history_len=1024,
                amax_compute_algo="max",
            )
            te_ctx_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
            print("⚡ Transformer Engine: FP8 (MXFP8 DelayedScaling)")

        elif PRECISION_MODE == "fp4":
            # NVFP4: E2M1 с 2-level block scaling
            # Работает ТОЛЬКО на Blackwell (B200/B100)
            # Включает: stochastic rounding, random Hadamard transform, 16x16 block scaling
            if hasattr(te_recipe, 'NVFP4BlockScaling'):
                fp4_recipe = te_recipe.NVFP4BlockScaling()
                te_ctx_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe)
                print("⚡ Transformer Engine: FP4 (NVFP4 BlockScaling)")
                print("   Stochastic rounding + Random Hadamard Transform")
            else:
                # Fallback: TE установлен но без NVFP4 (старая версия)
                print("⚠️  NVFP4BlockScaling не найден в transformer_engine!")
                print("   Нужен TE >= 2.7.0. Falling back to FP8...")
                fp8_recipe = te_recipe.DelayedScaling(
                    margin=0,
                    fp8_format=te_recipe.Format.HYBRID,
                    amax_history_len=1024,
                    amax_compute_algo="max",
                )
                te_ctx_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
                print("⚡ Transformer Engine: FP8 (fallback)")

    elif PRECISION_MODE in ("fp8", "fp4") and not TE_AVAILABLE:
        print(f"⚠️  PRECISION_MODE='{PRECISION_MODE}' но Transformer Engine не установлен!")
        print("   Установи: pip install transformer-engine[pytorch]")
        print("   Или используй NGC контейнер: nvcr.io/nvidia/pytorch:26.01-py3")
        print("   Продолжаю в bf16 режиме...")

    return device, dtype, amp_ctx, te_ctx_fn


# ============================================================
# ОЦЕНКА
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_dataset, val_dataset, config, device, amp_ctx, te_ctx_fn):
    """Оценка loss на train и val."""
    model.eval()
    losses = {}
    for name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        total = 0.0
        for _ in range(EVAL_SAMPLES):
            x, y = dataset.get_batch(BATCH_SIZE, device)
            # Два вложенных контекста: bf16 AMP + Transformer Engine (FP8/FP4)
            with amp_ctx:
                if te_ctx_fn is not None:
                    with te_ctx_fn():
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / EVAL_SAMPLES
    model.train()
    return losses


# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================

def train():
    device, dtype, amp_ctx, te_ctx_fn = setup()

    # --- Проверяем данные ---
    train_bin = DATA_DIR / "train.bin"
    val_bin = DATA_DIR / "val.bin"
    if not train_bin.exists() or not val_bin.exists():
        print("\n❌ Данные не подготовлены! Выполни сначала:")
        print("   python dataset.py")
        print("   (автоматически возьмёт ruwiki_full.txt и обученный токенизатор)")
        return

    # --- Загружаем токенизатор ---
    tokenizer = Tokenizer(str(TOKENIZER_PATH))
    print(f"📝 Токенизатор: {tokenizer.vocab_size} токенов")

    # --- Загружаем датасеты ---
    config = ModelConfig()
    config.vocab_size = tokenizer.vocab_size

    # Устанавливаем precision mode (bf16/fp8/fp4)
    # Это определяет, какие слои будут использовать te.Linear (FP8/FP4 ядра)
    config.precision_mode = PRECISION_MODE

    # Спецтокены, которые маскируются из loss (pad, endoftext, sep, mask)
    config.ignore_token_ids = [
        tid for tid in [tokenizer.pad_id, tokenizer.eot_id,
                        tokenizer.sep_id, tokenizer.mask_id]
        if tid >= 0  # -1 означает "не найден"
    ]
    print(f"   Маскируемые токены (не участвуют в loss): {config.ignore_token_ids}")

    # Gradient checkpointing — для 100M модели обычно не нужен (помещается в VRAM).
    # Также несовместим с Transformer Engine (FP8/FP4).
    use_te_mode = PRECISION_MODE in ("fp8", "fp4") and TE_AVAILABLE and device.type == "cuda"
    config.use_gradient_checkpointing = False
    print("📦 Gradient checkpointing: ВЫКЛ (100M модель помещается в VRAM)")

    train_dataset = MemmapDataset(str(train_bin), config.block_size)
    val_dataset = MemmapDataset(str(val_bin), config.block_size)
    print(f"📖 Train: {train_dataset.length:,} токенов | Val: {val_dataset.length:,} токенов")

    tokens_per_step = BATCH_SIZE * GRADIENT_ACCUM_STEPS * config.block_size
    print(f"📊 Токенов за шаг: {tokens_per_step:,} ({tokens_per_step/1e6:.2f}M)")
    total_tokens = tokens_per_step * MAX_STEPS
    print(f"📊 Всего токенов за обучение: {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    # --- Создаём модель ---
    model = MiniGPT(config).to(device)

    # Компилируем модель (PyTorch 2.0+) — ускорение ~20-30%
    # ВАЖНО:
    # - torch.compile конфликтует с Transformer Engine → отключаем при FP8/FP4
    # - torch.compile может вызвать segfault/OOM на Blackwell (RTX 50xx) →
    #   включаем только если COMPILE=1 задан явно
    # По умолчанию: ВЫКЛЮЧЕН. Для включения: COMPILE=1 python train.py
    use_te_mode = PRECISION_MODE in ("fp8", "fp4") and TE_AVAILABLE and device.type == "cuda"
    enable_compile = os.environ.get("COMPILE", "0") == "1"
    if use_te_mode:
        print("🔧 torch.compile отключён (Transformer Engine использует свои CUDA ядра)")
    elif enable_compile and hasattr(torch, 'compile') and device.type == "cuda":
        print("🔧 Компилирую модель (torch.compile)...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️  torch.compile не удался: {e}")
            print("   Продолжаю без компиляции")
    else:
        print("🔧 torch.compile отключён (для включения: COMPILE=1)")

    # --- Оптимизатор ---
    # Разделяем параметры: weight decay применяем только к матрицам весов,
    # НЕ к bias, нормализации и эмбеддингам.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # fused=True только на CUDA (ускоряет на ~5%, но не работает на CPU/MPS)
    use_fused = device.type == "cuda" and hasattr(torch.optim.AdamW, "fused")
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    print(f"🔧 Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"   Decay params: {sum(p.numel() for p in decay_params):,}")
    print(f"   No-decay params: {sum(p.numel() for p in no_decay_params):,}")

    # GradScaler для fp16 (не нужен для bf16)
    # При FP8/FP4: Transformer Engine сам управляет scaling-ом (amax/DelayedScaling),
    # поэтому отключаем GradScaler — он будет мешать TE loss scaling.
    use_grad_scaler = (dtype == torch.float16) and not use_te_mode
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    # --- Resume from checkpoint ---
    start_step = 0
    resume_path = CHECKPOINT_DIR / "klenskiy-gpt-100m-best.pt"
    if not resume_path.exists():
        # Ищем последний step_*
        steps = sorted(CHECKPOINT_DIR.glob("klenskiy-gpt-100m-step_*.pt"))
        if steps:
            resume_path = steps[-1]
        else:
            resume_path = None

    if resume_path and resume_path.exists():
        print(f"\n🔄 Найден чекпоинт: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        print(f"   Продолжаем с шага {start_step}")

    # --- Wandb ---
    if USE_WANDB:
        import wandb
        wandb.init(project=WANDB_PROJECT, config={
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "block_size": config.block_size,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRADIENT_ACCUM_STEPS,
            "lr": LEARNING_RATE,
            "max_steps": MAX_STEPS,
        })

    # --- Обучение ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🏋️  Начинаем обучение")
    print(f"   Шагов: {MAX_STEPS:,}")
    print(f"   Effective batch: {tokens_per_step:,} токенов")
    print(f"{'='*60}\n")

    model.train()
    start_time = time.time()
    running_loss = 0.0
    best_val_loss = float("inf")

    # --- CUDA Warmup ---
    # Первый forward pass на новом GPU (особенно Blackwell SM 12.0) запускает
    # JIT-компиляцию CUDA ядер. Это может занять 1-3 минуты.
    # Делаем warmup заранее чтобы не искажать таймеры обучения.
    print("⏳ CUDA warmup (компиляция ядер для GPU)...")
    warmup_start = time.time()
    with torch.no_grad():
        wx, wy = train_dataset.get_batch(BATCH_SIZE, device)
        with amp_ctx:
            if te_ctx_fn is not None:
                with te_ctx_fn():
                    _ = model(wx, wy)
            else:
                _ = model(wx, wy)
        del wx, wy
        torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f"✅ Warmup завершён за {warmup_time:.1f}s")
    start_time = time.time()  # Сбрасываем таймер после warmup

    for step in range(start_step, MAX_STEPS):
        step_start = time.time()

        # --- Learning Rate Schedule ---
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Gradient Accumulation ---
        # Вместо одного большого батча делаем несколько маленьких,
        # накапливаем градиенты, и обновляем веса один раз.
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRADIENT_ACCUM_STEPS):
            x, y = train_dataset.get_batch(BATCH_SIZE, device)

            # Два вложенных контекста:
            # 1. amp_ctx — torch.amp.autocast (bf16 базовый precision)
            # 2. te_ctx_fn() — Transformer Engine autocast (FP8/FP4 для te.Linear слоёв)
            # TE работает ПОВЕРХ bf16: активации bf16 → квантуются в FP8/FP4 → Tensor Core matmul
            with amp_ctx:
                if te_ctx_fn is not None:
                    with te_ctx_fn():
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                loss = loss / GRADIENT_ACCUM_STEPS  # нормализуем по кол-ву шагов

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            # Освобождаем ссылки на промежуточные тензоры между micro-steps
            del logits, loss

        # --- Gradient Clipping ---
        # Ограничиваем норму градиента. Без этого при 1B параметрах
        # один "плохой" батч может испортить все веса.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # --- Update ---
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss

        # --- Вывод первых шагов (чтобы видеть что обучение идёт) ---
        if step < 5 or (step < 50 and step % 10 == 0):
            step_time = time.time() - step_start
            print(f"  Шаг {step} | Loss: {accum_loss:.4f} | Время: {step_time:.2f}s | LR: {lr:.2e}")

        # --- Логирование ---
        if step % LOG_INTERVAL == 0 and step > 0:
            avg_loss = running_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            tokens_sec = step * tokens_per_step / elapsed
            eta_hours = (MAX_STEPS - step) / (step / elapsed) / 3600 if step > 0 else 0

            print(
                f"Шаг {step:6d}/{MAX_STEPS} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_sec:,.0f} | "
                f"ETA: {eta_hours:.1f}h"
            )

            if USE_WANDB:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/lr": lr,
                          "train/tokens_per_sec": tokens_sec}, step=step)

            running_loss = 0.0

        # --- Оценка ---
        if step % EVAL_INTERVAL == 0 and step > 0:
            losses = estimate_loss(model, train_dataset, val_dataset, config, device, amp_ctx, te_ctx_fn)
            print(
                f"  📊 Eval | Train: {losses['train']:.4f} | "
                f"Val: {losses['val']:.4f} | "
                f"PPL: {math.exp(losses['val']):.1f}"
            )

            if USE_WANDB:
                import wandb
                wandb.log({"eval/train_loss": losses["train"],
                          "eval/val_loss": losses["val"],
                          "eval/val_ppl": math.exp(losses["val"])}, step=step)

            # Сохраняем лучшую модель
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, optimizer, config, tokenizer, step, "best")
                print(f"  💾 Новый лучший val loss: {best_val_loss:.4f}")

        # --- Сэмпл ---
        if step % SAMPLE_INTERVAL == 0 and step > 0:
            model.eval()
            prompt = "В начале было"
            prompt_ids = tokenizer.encode(prompt)
            ctx_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            # Генерация: используем только amp_ctx (TE не нужен при инференсе —
            # веса уже в нужном формате, а FP8/FP4 autocast нужен только для обучения)
            with amp_ctx:
                generated = model.generate(ctx_tensor, max_new_tokens=200, temperature=0.8, top_k=50)
            text = tokenizer.decode(generated[0].tolist())
            print(f"\n  📝 Генерация (шаг {step}):")
            print(f"  {text[:500]}")
            print()
            model.train()

        # --- Периодический чекпоинт ---
        if step % SAVE_INTERVAL == 0 and step > 0:
            save_checkpoint(model, optimizer, config, tokenizer, step, f"step_{step}")

    # --- Финальное сохранение ---
    save_checkpoint(model, optimizer, config, tokenizer, MAX_STEPS, "final")
    print(f"\n🎉 Обучение завершено! Лучший val loss: {best_val_loss:.4f}")


def save_checkpoint(model, optimizer, config, tokenizer, step, name):
    """Сохраняет чекпоинт."""
    path = CHECKPOINT_DIR / f"klenskiy-gpt-100m-{name}.pt"

    # Если модель скомпилирована, берём оригинал
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "step": step,
        "tokenizer_path": str(TOKENIZER_PATH),
    }, path)
    print(f"  💾 Чекпоинт: {path}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        apply_test_config()
        print("🧪" + "=" * 58)
        print("🧪  ТЕСТОВЫЙ РЕЖИМ")
        print("🧪  Шагов: 500, Batch: 8×4×2048 = 65K tok/step")
        print("🧪  Precision: bf16 (для совместимости)")
        print("🧪  Убедись что данные готовы: python dataset.py --test")
        print("🧪" + "=" * 58)
    train()
