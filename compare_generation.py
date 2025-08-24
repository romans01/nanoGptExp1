#!/usr/bin/env python3
"""
Скрипт для сравнения качества генерации разных моделей с центрированием
"""

import os
import subprocess
import sys

def test_model_generation(model_dir, model_name, prompt="ROMEO:", max_tokens=150):
    """Тестирует генерацию одной модели"""
    
    if not os.path.exists(f"{model_dir}/ckpt.pt"):
        print(f"❌ {model_name}: Checkpoint не найден")
        return None
    
    print(f"\n🎭 {model_name.upper()}")
    print("=" * 60)
    
    cmd = [
        'python', 'sample_centered.py',
        f'--out_dir={model_dir}',
        '--num_samples=1',
        f'--max_new_tokens={max_tokens}',
        f'--start={prompt}',
        '--temperature=0.8',
        '--top_k=200'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Извлекаем сгенерированный текст
            output_lines = result.stdout.split('\n')
            
            # Находим начало и конец генерации
            start_found = False
            generated_text = []
            
            for line in output_lines:
                if prompt in line and not start_found:
                    start_found = True
                    generated_text.append(line)
                elif start_found and '---------------' in line:
                    break
                elif start_found:
                    generated_text.append(line)
            
            if generated_text:
                print('\n'.join(generated_text))
                return '\n'.join(generated_text)
            else:
                print("Не удалось извлечь сгенерированный текст")
                return None
        else:
            print(f"Ошибка генерации: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Таймаут при генерации")
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def main():
    print("🎭 Сравнение качества генерации моделей с центрированием")
    print("=" * 70)
    
    # Модели для тестирования
    models = [
        ("out-centering-bpe-baseline_bpe", "Baseline (без центрирования)"),
        ("out-centering-bpe-qk_centered_bpe", "QK Centered (центрирование query/key)"),
        ("out-centering-bpe-final_centered_bpe", "Final Centered (центрирование финальных эмбеддингов)"),
        ("out-centering-bpe-full_centered_bpe", "Full Centered (полное центрирование)"),
        ("out-centering-bpe-block_centered_bpe", "Block Centered (центрирование выходов блоков)"),
    ]
    
    # Разные промпты для тестирования
    prompts = [
        "ROMEO:",
        "JULIET:",
        "To be or not to be",
        "HAMLET:",
        "What light through yonder window"
    ]
    
    # Выбираем промпт
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = prompts[0]  # По умолчанию ROMEO:
    
    print(f"🎯 Промпт: '{prompt}'")
    print(f"📝 Длина генерации: 150 токенов")
    
    results = {}
    
    # Тестируем все модели
    for model_dir, model_name in models:
        generated_text = test_model_generation(model_dir, model_name, prompt)
        if generated_text:
            results[model_name] = generated_text
    
    # Краткое сравнение
    print(f"\n{'='*70}")
    print("📊 КРАТКОЕ СРАВНЕНИЕ:")
    print(f"{'='*70}")
    
    for model_name, text in results.items():
        lines = text.split('\n')
        first_lines = lines[:3]  # Первые 3 строки
        preview = ' '.join(first_lines).replace('\n', ' ').strip()
        if len(preview) > 80:
            preview = preview[:77] + "..."
        
        print(f"\n🎭 {model_name}:")
        print(f"   {preview}")
    
    print(f"\n✅ Протестировано моделей: {len(results)}")
    
    if len(results) > 0:
        print(f"\n💡 Для тестирования с другим промптом:")
        print(f"   python {sys.argv[0]} \"JULIET:\"")
        print(f"   python {sys.argv[0]} \"To be or not to be\"")

if __name__ == "__main__":
    main()
