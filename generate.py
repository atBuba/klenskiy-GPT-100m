"""
✨ Генерация текста моделью klenskiy-GPT-1B

Использование:
  python generate.py --prompt "Однажды" --length 500 --temperature 0.8
  python generate.py --interactive  # интерактивный режим (чат)
"""

import os
import argparse
import torch
from pathlib import Path

from model import MiniGPT, ModelConfig
from tokenizer import Tokenizer


def load_model(checkpoint_path, device, rope_scale_factor=None):
    """
    Загружает модель из чекпоинта.

    rope_scale_factor: если задан, включает NTK-aware RoPE scaling
        для расширения контекста (например, 2.0 → 4096, 4.0 → 8192).
        Это работает БЕЗ переобучения — просто меняет частоты RoPE.
    """
    print(f"📦 Загружаю модель: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    tokenizer_path = checkpoint.get("tokenizer_path")

    # Применяем RoPE scaling для длинных контекстов
    if rope_scale_factor and rope_scale_factor > 1.0:
        config.rope_scaling = {"type": "ntk", "factor": rope_scale_factor}
        new_max = int(config.block_size * rope_scale_factor)
        print(f"   🔄 NTK RoPE scaling: {config.block_size} → {new_max} токенов")

    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    step = checkpoint.get("step", "?")
    print(f"   Шаг обучения: {step}")

    return model, config, tokenizer_path


def generate(model, tokenizer, config, device, prompt, max_tokens, temperature, top_k, top_p):
    """Генерирует текст."""
    # Кодируем промпт
    if prompt:
        input_ids = tokenizer.encode(prompt)
    else:
        input_ids = [tokenizer.bos_id]

    context = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Генерация
    with torch.no_grad():
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.generate(context, max_new_tokens=max_tokens,
                                       temperature=temperature, top_k=top_k)
        else:
            output = model.generate(context, max_new_tokens=max_tokens,
                                   temperature=temperature, top_k=top_k)

    return tokenizer.decode(output[0].tolist())


def interactive_mode(model, tokenizer, config, device, temperature, top_k, top_p):
    """Интерактивный режим — вводишь промпт, получаешь продолжение."""
    print("\n" + "=" * 60)
    print("💬 Интерактивный режим (введи 'q' для выхода)")
    print(f"   temperature={temperature}, top_k={top_k}")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n📝 Промпт: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("q", "quit", "exit"):
            break

        if not prompt:
            continue

        text = generate(model, tokenizer, config, device, prompt,
                       max_tokens=300, temperature=temperature,
                       top_k=top_k, top_p=top_p)

        print(f"\n🤖 {text}")


def main():
    parser = argparse.ArgumentParser(description="Генерация текста klenskiy-GPT-1B")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Путь к чекпоинту (default: best)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Начальный текст")
    parser.add_argument("--length", type=int, default=500,
                        help="Длина генерации в токенах")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k фильтрация")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--interactive", action="store_true",
                        help="Интерактивный режим")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Сколько примеров сгенерировать")
    parser.add_argument("--rope_scale", type=float, default=None,
                        help="NTK RoPE scale factor (2.0=4096, 4.0=8192 контекст)")
    args = parser.parse_args()

    # Устройство
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Чекпоинт
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Ищем best, потом final, потом последний step_*
        for name in ["klenskiy-gpt-100m-best.pt", "klenskiy-gpt-100m-final.pt"]:
            p = checkpoint_dir / name
            if p.exists():
                checkpoint_path = str(p)
                break
        else:
            # Последний step_*
            steps = sorted(checkpoint_dir.glob("klenskiy-gpt-100m-step_*.pt"))
            if steps:
                checkpoint_path = str(steps[-1])
            else:
                print("❌ Чекпоинт не найден! Сначала обучи модель: python train.py")
                return

    # Загрузка
    model, config, tokenizer_path = load_model(checkpoint_path, device, args.rope_scale)
    tokenizer = Tokenizer(tokenizer_path)

    if args.interactive:
        interactive_mode(model, tokenizer, config, device,
                        args.temperature, args.top_k, args.top_p)
    else:
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"\n{'='*60}")
                print(f"📝 Пример {i+1}/{args.num_samples}")

            text = generate(model, tokenizer, config, device,
                          args.prompt, args.length, args.temperature,
                          args.top_k, args.top_p)
            print(f"\n{text}")


if __name__ == "__main__":
    main()
