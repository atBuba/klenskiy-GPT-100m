"""
🧠 Шаг 2: Архитектура модели — Modern Transformer (Llama-style)

ОБНОВЛЁННАЯ ВЕРСИЯ с современными техниками из Llama 2/3, Mistral, Qwen:

Что изменилось (OLD → NEW):
┌─────────────────────┬──────────────────────┬────────────────────────────┐
│ Компонент           │ Было (ванильный)     │ Стало (modern)             │
├─────────────────────┼──────────────────────┼────────────────────────────┤
│ Позиции             │ Learned Positional   │ RoPE (Rotary Position      │
│                     │ Embedding            │ Embedding)                 │
├─────────────────────┼──────────────────────┼────────────────────────────┤
│ Attention           │ Multi-Head (MHA)     │ Grouped Query Attention    │
│                     │ 12 голов × свои QKV  │ (GQA) — K,V разделяются    │
├─────────────────────┼──────────────────────┼────────────────────────────┤
│ Feed-Forward        │ Linear → GELU →      │ SwiGLU — gated activation  │
│                     │ Linear               │ с тремя матрицами          │
├─────────────────────┼──────────────────────┼────────────────────────────┤
│ Нормализация        │ LayerNorm            │ RMSNorm (быстрее,          │
│                     │                      │ без вычитания среднего)     │
└─────────────────────┴──────────────────────┴────────────────────────────┘

💡 ЗАМЕТКА ТИМЛИДА:
Эта архитектура — по сути mini-Llama. Те же решения, что используют
Meta (Llama), Mistral AI, Google (Gemma). Разница только в масштабе.
Когда ты разберёшься в этом коде, ты сможешь прочитать код реальных
open-source моделей и понять каждую строку.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import math

# ============================================================
# NVIDIA Transformer Engine (опционально)
# ============================================================
# Если установлен — используем te.Linear для FP8/FP4 на Blackwell/Hopper.
# Если нет — fallback на обычный nn.Linear (bf16/fp32).
#
# Установка: pip install transformer-engine[pytorch]
# Или через NGC контейнер: nvcr.io/nvidia/pytorch:26.01-py3

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


# ============================================================
# КОНФИГУРАЦИЯ МОДЕЛИ
# ============================================================

class ModelConfig:
    block_size: int = 2048     # Длина контекста для обучения

    n_embd: int = 768          # Размер эмбеддинга (hidden dimension)

    n_head: int = 12           # Количество Query-голов
    n_kv_head: int = 4         # Количество KV-голов (для GQA)
                               # n_head / n_kv_head = 3 → каждая KV-голова обслуживает 3 Q-головы

    n_layer: int = 12          # Количество трансформер-блоков

    dropout: float = 0.0       # При большом датасете dropout не нужен (данных достаточно)

    vocab_size: int = 32000    # BPE словарь (SentencePiece)

    # Список ID спецтокенов, для которых НЕ считаем loss при обучении
    # (pad, endoftext, sep, mask). Заполняется из токенизатора в train.py
    ignore_token_ids: list = None

    # RoPE scaling для расширения контекста на инференсе
    rope_scaling: dict = None

    # Gradient checkpointing — пересчитываем активации вместо хранения
    # Для 100M модели обычно не нужен (помещается в VRAM даже с большим batch)
    use_gradient_checkpointing: bool = False

    # Precision mode для обучения
    precision_mode: str = "bf16"
    fp4_high_precision_layers: int = 2

    # Формула для ~100M параметров:
    # Embedding:  vocab_size × n_embd = 32K × 768 = 24.6M
    # Per layer:  attention(~1.6M) + FFN(~4.7M) = ~6.3M
    # 12 layers:  12 × 6.3M ≈ 75.6M
    # Total:      24.6M + 75.6M ≈ 100M

    # SwiGLU intermediate size: 8/3 × n_embd, округлённый до кратного 256
    @property
    def intermediate_size(self):
        raw = int(8 / 3 * self.n_embd)
        return ((raw + 255) // 256) * 256  # Округляем до кратного 256 (лучше для GPU)

    @property
    def max_seq_len(self):
        """Максимальная длина с учётом RoPE scaling (для предвычисления частот)."""
        if self.rope_scaling and self.rope_scaling.get("type") == "ntk":
            return int(self.block_size * self.rope_scaling["factor"])
        return self.block_size


# ============================================================
# Адаптивный Linear: nn.Linear или te.Linear
# ============================================================
# При FP8/FP4 режиме используем te.Linear — он автоматически
# квантует веса и активации в нужный формат при forward pass
# внутри te.fp8_autocast() контекста.

def make_linear(in_features, out_features, bias=False, use_te=False):
    """
    Создаёт Linear слой: te.Linear для FP8/FP4 или nn.Linear для bf16.

    te.Linear автоматически:
    - Квантует веса и активации в FP8/FP4 при forward
    - Использует Tensor Core ядра для максимальной производительности
    - Выполняет de-quantize при backward
    """
    if use_te and TE_AVAILABLE:
        return te.Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


# ============================================================
# RMSNorm (Root Mean Square Layer Normalization)
# ============================================================
# Упрощённая версия LayerNorm: убираем вычитание среднего.
# Зачем? Экспериментально показано (Zhang & Sennrich, 2019), что
# re-centering (вычитание μ) не несёт пользы для трансформеров,
# а re-scaling (деление на σ) — несёт. RMSNorm оставляет только полезную часть.
#
# LayerNorm:  γ * (x - μ) / √(σ² + ε) + β
# RMSNorm:    γ * x / √(mean(x²) + ε)          ← проще и быстрее
#
# Используется в: Llama, Llama 2, Llama 3, Mistral, Gemma

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ — обучаемый масштаб

    def forward(self, x):
        # x.shape = (B, T, dim)
        # rsqrt = 1/sqrt — одна fused операция вместо sqrt + div (быстрее на Tensor Cores)
        # .float() → считаем в FP32 для численной стабильности (как в Llama reference)
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


# ============================================================
# RoPE (Rotary Position Embedding)
# ============================================================
# Вместо прибавления позиционного вектора к эмбеддингу,
# RoPE ВРАЩАЕТ вектора Q и K в зависимости от их позиции.
#
# Ключевая идея: скалярное произведение Q·K после вращения
# зависит только от ОТНОСИТЕЛЬНОГО расстояния между позициями,
# а не от абсолютных позиций. Это то, что нам нужно для языка:
# "кот ел рыбу" — расстояние между "кот" и "ел" = 1, и это
# должно работать одинаково в начале и в конце текста.
#
# Математика: представляем вектор как набор 2D-точек (пар чисел)
# и вращаем каждую пару на угол θ * позиция, где θ разный для
# каждой пары (частоты: θ_i = 1 / 10000^(2i/d)).
#
# Используется в: Llama, Llama 2/3, Mistral, Qwen, Gemma, CodeLlama

def precompute_rope_frequencies(head_dim, max_seq_len, theta=10000.0, rope_scaling=None):
    """
    Предвычисляет sin/cos для RoPE.

    head_dim: размер одной головы (d_k)
    max_seq_len: максимальная длина последовательности
    theta: базовая частота (10000 — стандарт из оригинальной статьи)
    rope_scaling: dict с параметрами NTK-aware scaling (опционально)
        {"type": "ntk", "factor": 2.0} — расширяет контекст в factor раз

    NTK-aware RoPE scaling (Neural Tangent Kernel):
    Вместо простой интерполяции позиций (которая портит локальные связи),
    NTK увеличивает базовую частоту θ: θ' = θ × factor^(d/(d-2))
    Это позволяет расширить контекст (например, 2048 → 4096 или 8192)
    БЕЗ переобучения, с минимальной потерей качества.

    Используется в: CodeLlama, Llama 3, Yi, Mistral (длинные контексты)
    Статья: "NTK-Aware Scaled RoPE" (Reddit/bloc97, 2023)
    """
    # NTK-aware scaling: увеличиваем theta вместо интерполяции позиций
    if rope_scaling is not None and rope_scaling.get("type") == "ntk":
        factor = rope_scaling["factor"]
        # θ' = θ × factor^(dim / (dim - 2))
        theta = theta * (factor ** (head_dim / (head_dim - 2)))

    # Частоты: θ_i = 1 / θ^(2i/d) для i = 0, 1, ..., d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Позиции: 0, 1, 2, ..., max_seq_len-1
    positions = torch.arange(max_seq_len).float()

    # Углы вращения: каждая позиция × каждая частота
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)

    # Возвращаем cos и sin (нужны оба для вращения)
    return torch.cos(angles), torch.sin(angles)  # каждый: (max_seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    """
    Применяет RoPE к тензору x.

    x.shape = (B, n_heads, T, head_dim)
    cos, sin shape = (T, head_dim/2)

    Для каждой пары (x[2i], x[2i+1]) применяем вращение на угол:
    x'[2i]   = x[2i] * cos - x[2i+1] * sin
    x'[2i+1] = x[2i] * sin + x[2i+1] * cos
    """
    B, H, T, D = x.shape

    # Разбиваем вектор на пары
    x_even = x[..., 0::2]   # чётные индексы: (B, H, T, D/2)
    x_odd = x[..., 1::2]    # нечётные индексы: (B, H, T, D/2)

    # Подгоняем размеры cos/sin для broadcasting
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)

    # Применяем вращение
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # Собираем обратно, чередуя чётные и нечётные
    out = torch.stack([out_even, out_odd], dim=-1)  # (B, H, T, D/2, 2)
    return out.reshape(B, H, T, D)


# ============================================================
# Grouped Query Attention (GQA)
# ============================================================
# Эволюция Multi-Head Attention:
#
# MHA (оригинал):  12 Q-голов, 12 K-голов, 12 V-голов  (все разные)
# MQA:             12 Q-голов,  1 K-голова,  1 V-голова  (все Q делят одну KV)
# GQA (наш):       12 Q-голов,  4 K-головы,  4 V-головы  (группы по 3 Q на 1 KV)
#
# Зачем? При генерации (инференсе) нужно хранить K и V для всех
# предыдущих токенов — это "KV-кэш". С GQA кэш в 3 раза меньше,
# а качество почти не страдает.
#
# Используется в: Llama 2 70B, Llama 3, Mistral, Gemma

class GroupedQueryAttention(nn.Module):
    def __init__(self, config, use_te=False):
        super().__init__()

        self.n_head = config.n_head         # 12 Q-голов
        self.n_kv_head = config.n_kv_head   # 4 KV-головы
        self.n_rep = self.n_head // self.n_kv_head  # 3 — коэффициент повторения
        self.head_dim = config.n_embd // config.n_head  # 64

        # Q проекция — для всех 16 голов
        self.wq = make_linear(config.n_embd, config.n_head * self.head_dim, use_te=use_te)
        # K и V проекции — только для 4 KV-голов (в 4 раза меньше параметров!)
        self.wk = make_linear(config.n_embd, config.n_kv_head * self.head_dim, use_te=use_te)
        self.wv = make_linear(config.n_embd, config.n_kv_head * self.head_dim, use_te=use_te)

        # Выходная проекция
        self.wo = make_linear(config.n_head * self.head_dim, config.n_embd, use_te=use_te)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Каузальная маска для fallback (PyTorch < 2.0)
        # SDPA с is_causal=True НЕ использует этот буфер — он нужен только для fallback.
        # Для max_seq_len=2048: буфер = 2048²×4 bytes = 16MB. Лениво создаём только при fallback.
        self._mask = None
        self._max_seq_len = config.max_seq_len

        # Предвычисляем RoPE частоты (с учётом NTK scaling для длинных контекстов)
        rope_cos, rope_sin = precompute_rope_frequencies(
            self.head_dim, config.max_seq_len, rope_scaling=config.rope_scaling
        )
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    def forward(self, x):
        B, T, C = x.shape

        # Проецируем в Q, K, V
        q = self.wq(x)  # (B, T, n_head * head_dim)
        k = self.wk(x)  # (B, T, n_kv_head * head_dim) — меньше!
        v = self.wv(x)  # (B, T, n_kv_head * head_dim) — меньше!

        # Reshape в головы
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, 12, T, 16)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, 4, T, 16)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, 4, T, 16)

        # Применяем RoPE к Q и K (не к V!)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Повторяем K и V для каждой группы Q-голов
        # (B, 4, T, 128) → (B, 16, T, 128)
        # expand + reshape вместо repeat_interleave:
        # expand создаёт VIEW без копирования данных (O(1) память),
        # repeat_interleave создаёт физическую копию (O(n) память).
        # На Blackwell (1.79 TB/s bandwidth) это критично — меньше данных перемещать.
        k = k[:, :, None, :, :].expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)

        # Scaled Dot-Product Attention
        # PyTorch 2.0+ F.scaled_dot_product_attention автоматически использует
        # FlashAttention или Memory-Efficient Attention — в 2-4x быстрее!
        if hasattr(F, 'scaled_dot_product_attention'):
            # SDPA автоматически применяет каузальную маску и dropout
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,  # каузальная маска встроена
            )
        else:
            # Fallback для PyTorch < 2.0
            if self._mask is None or self._mask.size(0) < T:
                self._mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len, device=x.device))
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self._mask[:T, :T] == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = weights @ v  # (B, n_head, T, head_dim)

        # Собираем головы обратно
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, 192)

        # Выходная проекция
        return self.resid_dropout(self.wo(out))


# ============================================================
# SwiGLU Feed-Forward Network
# ============================================================
# Замена стандартного FFN на gated version:
#
# Старый FFN:     GELU(x·W₁) · W₂
# SwiGLU:         (Swish(x·W_gate) ⊙ x·W_up) · W_down
#
# ⊙ — поэлементное умножение (gate "решает", что пропустить)
# Swish(x) = x · σ(x) — smooth activation, похожая на ReLU но плавнее
#
# Три матрицы вместо двух, но intermediate_size меньше (8/3 вместо 4),
# так что общее кол-во параметров примерно то же.
#
# Используется в: Llama, Llama 2/3, PaLM, Mistral, Gemma

class SwiGLUFeedForward(nn.Module):
    def __init__(self, config, use_te=False):
        super().__init__()
        hidden = config.intermediate_size  # 2048 для n_embd=768

        # Три проекции вместо двух
        self.w_gate = make_linear(config.n_embd, hidden, use_te=use_te)  # для gate (Swish)
        self.w_up = make_linear(config.n_embd, hidden, use_te=use_te)    # для данных
        self.w_down = make_linear(hidden, config.n_embd, use_te=use_te)  # проекция обратно
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: Swish(x·W_gate) ⊙ (x·W_up), затем проекция вниз
        gate = F.silu(self.w_gate(x))   # Swish/SiLU activation
        up = self.w_up(x)               # линейная проекция
        return self.dropout(self.w_down(gate * up))  # gated + проекция вниз


# ============================================================
# TRANSFORMER BLOCK (Modern)
# ============================================================
# Pre-norm с RMSNorm + GQA + SwiGLU + Residual

class TransformerBlock(nn.Module):
    def __init__(self, config, use_te=False):
        super().__init__()
        self.attention = GroupedQueryAttention(config, use_te=use_te)
        self.feed_forward = SwiGLUFeedForward(config, use_te=use_te)
        # При TE используем te.RMSNorm (fused kernel, быстрее)
        if use_te and TE_AVAILABLE:
            self.ln1 = te.RMSNorm(config.n_embd, eps=1e-6)
            self.ln2 = te.RMSNorm(config.n_embd, eps=1e-6)
        else:
            self.ln1 = RMSNorm(config.n_embd)
            self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x):
        # Pre-norm + residual (как и раньше, но с RMSNorm)
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


# ============================================================
# ПОЛНАЯ МОДЕЛЬ (Modern MiniGPT)
# ============================================================

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token Embedding (без Position Embedding — позиции теперь через RoPE!)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Dropout на эмбеддингах
        self.embed_dropout = nn.Dropout(config.dropout)

        # Стек трансформер-блоков
        # При FP8/FP4: используем te.Linear для квантованных вычислений
        # При FP4: последние N слоёв работают в FP8 (более чувствительны к квантизации)
        use_te_mode = config.precision_mode in ("fp8", "fp4") and TE_AVAILABLE
        blocks = []
        for i in range(config.n_layer):
            # Для FP4: последние fp4_high_precision_layers слоёв остаются на FP8
            # (они будут в TE, но train.py задаст им FP8 рецепт вместо FP4)
            layer_use_te = use_te_mode
            blocks.append(TransformerBlock(config, use_te=layer_use_te))
        self.blocks = nn.ModuleList(blocks)
        self._te_enabled = use_te_mode

        # Финальная нормализация (RMSNorm)
        if use_te_mode:
            self.ln_final = te.RMSNorm(config.n_embd, eps=1e-6)
        else:
            self.ln_final = RMSNorm(config.n_embd)

        # Output Head: ВСЕГДА nn.Linear (не квантуем — output layer чувствителен)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: output_head делит веса с token_embedding
        # Это экономит параметры и улучшает качество — идея в том, что
        # "понимание" символа (embedding) и "предсказание" символа (output)
        # используют одно и то же представление.
        self.output_head.weight = self.token_embedding.weight

        # Инициализация весов
        self.apply(self._init_weights)
        self._apply_scaled_init()  # Scaled init для residual-проекций

        n_params = sum(p.numel() for p in self.parameters())
        te_status = f" + TE({config.precision_mode.upper()})" if self._te_enabled else ""
        print(f"🧠 Модель создана: {n_params:,} параметров")
        print(f"   Архитектура: RoPE + GQA({config.n_head}q/{config.n_kv_head}kv) + SwiGLU + RMSNorm{te_status}")
        print(f"   Hidden: {config.n_embd}, Intermediate: {config.intermediate_size}, Layers: {config.n_layer}")
        if self._te_enabled and config.precision_mode == "fp4":
            print(f"   FP4: последние {config.fp4_high_precision_layers} слоёв в FP8 (higher precision)")

    def _init_weights(self, module):
        """
        Инициализация весов (важно для стабильного обучения!)

        Scaled init: выходные проекции attention (wo) и FFN (w_down)
        масштабируются на 1/√(2·n_layer). Это стабилизирует обучение
        глубоких моделей — без этого residual stream "взрывается".
        Используется в GPT-2, Llama, и большинстве современных LLM.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif TE_AVAILABLE and isinstance(module, te.Linear):
            # te.Linear — НЕ подкласс nn.Linear, инициализируем отдельно
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_scaled_init(self):
        """Применяет scaled init к residual-проекциям после основной инициализации."""
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layer)
        for block in self.blocks:
            # Выходная проекция attention (nn.Linear или te.Linear — оба имеют .weight)
            if hasattr(block.attention.wo, 'weight'):
                torch.nn.init.normal_(block.attention.wo.weight, mean=0.0, std=0.02 * scale)
            # Выходная проекция FFN
            if hasattr(block.feed_forward.w_down, 'weight'):
                torch.nn.init.normal_(block.feed_forward.w_down.weight, mean=0.0, std=0.02 * scale)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. Token embedding (позиции кодируются через RoPE в attention)
        x = self.token_embedding(idx)  # (B, T, n_embd)
        x = self.embed_dropout(x)

        # 2. Трансформер-блоки
        # Gradient checkpointing: вместо хранения всех активаций в VRAM,
        # пересчитываем их при backward pass. Экономит ~60% памяти,
        # стоит ~30% скорости. Включается через config.use_gradient_checkpointing
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # 3. Финальная нормализация
        x = self.ln_final(x)

        # 4. Логиты
        logits = self.output_head(x)  # (B, T, vocab_size)

        # 5. Loss с маскированием спецтокенов
        # Спецтокены (pad, endoftext, sep, mask) не несут языковой информации —
        # нет смысла учить модель их предсказывать. Маскируем их из loss.
        #
        # ОПТИМИЗАЦИЯ: используем ignore_index=-100 вместо boolean indexing.
        # Boolean indexing (flat_logits[mask]) создаёт копии тензоров и ломает
        # torch.compile (динамические формы). ignore_index работает in-place
        # внутри CUDA ядра cross_entropy — быстрее и совместимо с компиляцией.
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B * T, C)
            flat_targets = targets.view(B * T)

            if self.config.ignore_token_ids:
                # Заменяем спецтокены на -100 (стандартный ignore_index PyTorch)
                # clone() нужен чтобы не менять оригинальные targets
                masked_targets = flat_targets.clone()
                for tid in self.config.ignore_token_ids:
                    masked_targets[masked_targets == tid] = -100
                loss = F.cross_entropy(flat_logits, masked_targets, ignore_index=-100)
            else:
                loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Авторегрессивная генерация текста."""
        # С NTK RoPE scaling контекст может быть длиннее block_size
        max_ctx = self.config.max_seq_len
        for _ in range(max_new_tokens):
            # Обрезаем контекст до максимальной длины
            idx_cond = idx[:, -max_ctx:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k фильтрация
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Сэмплинг
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
