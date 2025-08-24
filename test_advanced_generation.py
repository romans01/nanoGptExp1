#!/usr/bin/env python3
"""
Тестирование генерации текста для лучших моделей с расширенным центрированием
"""

import os
import torch
import subprocess
from pathlib import Path

def test_model_generation(model_path, model_name, prompts, max_new_tokens=200):
    """Тестирует генерацию для одной модели"""
    
    print(f"\n🎭 {model_name.upper()}:")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return None
    
    results = {}
    
    for prompt in prompts:
        print(f"\n🎯 Промпт: '{prompt}'")
        print("=" * 30)
        
        try:
            # Используем sample_centered.py для центрированных моделей
            if 'baseline' in model_name.lower():
                sample_script = 'sample.py'
            else:
                sample_script = 'sample_centered.py'
            
            cmd = [
                'python', sample_script,
                '--out_dir', model_path,
                '--start', prompt,
                '--num_samples', '1',
                '--max_new_tokens', str(max_new_tokens),
                '--temperature', '0.8',
                '--top_k', '200'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Извлекаем сгенерированный текст
                output_lines = result.stdout.strip().split('\n')
                
                # Ищем строки после "No meta.pkl found" или похожих
                generation_started = False
                generated_text = []
                
                for line in output_lines:
                    if prompt in line and not generation_started:
                        generation_started = True
                        generated_text.append(line)
                    elif generation_started:
                        generated_text.append(line)
                
                if generated_text:
                    full_text = '\n'.join(generated_text)
                    print(full_text)
                    results[prompt] = full_text
                else:
                    print("⚠️  Не удалось извлечь сгенерированный текст")
                    results[prompt] = "Ошибка извлечения"
            else:
                print(f"❌ Ошибка генерации: {result.stderr[:200]}")
                results[prompt] = f"Ошибка: {result.stderr[:100]}"
                
        except subprocess.TimeoutExpired:
            print("⏱️  Таймаут генерации")
            results[prompt] = "Таймаут"
        except Exception as e:
            print(f"❌ Исключение: {e}")
            results[prompt] = f"Исключение: {str(e)}"
    
    return results

def compare_generations():
    """Сравнивает генерации лучших моделей"""
    
    print("🎭 СРАВНЕНИЕ ГЕНЕРАЦИИ ЛУЧШИХ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Модели для тестирования (лучшие по результатам 1K экспериментов)
    models = [
        {
            'path': 'out-advanced-1k-baseline_1k',
            'name': 'Baseline 1K',
            'description': 'Контрольная группа'
        },
        {
            'path': 'out-advanced-1k-embeddings_centered_1k', 
            'name': 'Embeddings Centered 1K',
            'description': '🥇 ПОБЕДИТЕЛЬ (+2.26%)'
        },
        {
            'path': 'out-advanced-1k-qk_plus_value_1k',
            'name': 'QK + Value 1K', 
            'description': '🥈 2-е место (+1.89%)'
        },
        {
            'path': 'out-advanced-1k-value_centered_1k',
            'name': 'Value Centered 1K',
            'description': 'Value центрирование (+1.13%)'
        }
    ]
    
    # Тестовые промпты
    prompts = [
        "ROMEO:",
        "JULIET:", 
        "To be or not to be",
        "Once upon a time",
        "The meaning of life"
    ]
    
    all_results = {}
    
    # Тестируем каждую модель
    for model in models:
        print(f"\n{'='*60}")
        print(f"🚀 Тестируем: {model['name']}")
        print(f"📝 Описание: {model['description']}")
        print(f"📁 Путь: {model['path']}")
        
        results = test_model_generation(
            model['path'], 
            model['name'], 
            prompts,
            max_new_tokens=150
        )
        
        if results:
            all_results[model['name']] = results
    
    # Создаем сравнительный отчет
    create_comparison_report(all_results, prompts)
    
    return all_results

def create_comparison_report(all_results, prompts):
    """Создает сравнительный отчет генераций"""
    
    print(f"\n\n📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ГЕНЕРАЦИЙ")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\n🎯 ПРОМПТ: '{prompt}'")
        print("=" * 40)
        
        for model_name, results in all_results.items():
            if prompt in results:
                print(f"\n🎭 {model_name}:")
                print("-" * 25)
                
                generation = results[prompt]
                if len(generation) > 300:
                    generation = generation[:300] + "..."
                
                print(generation)
    
    # Сохраняем отчет в файл
    save_generation_report(all_results, prompts)

def save_generation_report(all_results, prompts):
    """Сохраняет отчет в файл"""
    
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"training_plots/generation_comparison_{timestamp}.txt"
    
    os.makedirs('training_plots', exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🎭 СРАВНЕНИЕ ГЕНЕРАЦИИ ЛУЧШИХ МОДЕЛЕЙ\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ:\n")
        f.write("🥇 Embeddings Centered: +2.26% улучшение\n")
        f.write("🥈 QK + Value: +1.89% улучшение\n") 
        f.write("🥉 Value Centered: +1.13% улучшение\n")
        f.write("📊 Baseline: контрольная группа\n\n")
        
        for prompt in prompts:
            f.write(f"\n🎯 ПРОМПТ: '{prompt}'\n")
            f.write("=" * 50 + "\n")
            
            for model_name, results in all_results.items():
                if prompt in results:
                    f.write(f"\n🎭 {model_name}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{results[prompt]}\n")
        
        f.write(f"\n\n📝 Отчет создан: {timestamp}\n")
        f.write("🎯 Все модели обучены на 1000 итераций\n")
        f.write("📊 Температура генерации: 0.8\n")
        f.write("🔢 Максимум токенов: 150\n")
    
    print(f"\n💾 Отчет сохранен: {report_path}")

def analyze_generation_quality(all_results):
    """Анализирует качество генераций"""
    
    print(f"\n🔍 АНАЛИЗ КАЧЕСТВА ГЕНЕРАЦИЙ")
    print("=" * 40)
    
    quality_metrics = {}
    
    for model_name, results in all_results.items():
        metrics = {
            'avg_length': 0,
            'shakespeare_style': 0,
            'coherence': 0,
            'errors': 0
        }
        
        valid_generations = 0
        
        for prompt, generation in results.items():
            if "Ошибка" not in generation and "Таймаут" not in generation:
                valid_generations += 1
                
                # Длина генерации
                metrics['avg_length'] += len(generation)
                
                # Стиль Шекспира (простая эвристика)
                shakespeare_words = ['thou', 'thee', 'thy', 'hath', 'doth', 'art', 'shall', 'whence', 'wherefore']
                shakespeare_score = sum(1 for word in shakespeare_words if word.lower() in generation.lower())
                metrics['shakespeare_style'] += shakespeare_score
                
                # Связность (отсутствие повторов)
                words = generation.split()
                if len(words) > 10:
                    unique_ratio = len(set(words)) / len(words)
                    metrics['coherence'] += unique_ratio
            else:
                metrics['errors'] += 1
        
        if valid_generations > 0:
            metrics['avg_length'] /= valid_generations
            metrics['shakespeare_style'] /= valid_generations
            metrics['coherence'] /= valid_generations
        
        quality_metrics[model_name] = metrics
        
        print(f"\n📊 {model_name}:")
        print(f"  📏 Средняя длина: {metrics['avg_length']:.0f} символов")
        print(f"  🎭 Стиль Шекспира: {metrics['shakespeare_style']:.1f} слов")
        print(f"  🔗 Связность: {metrics['coherence']:.2f}")
        print(f"  ❌ Ошибки: {metrics['errors']}")
    
    return quality_metrics

def main():
    print("🎭 Запуск тестирования генерации для лучших моделей")
    print("🎯 Цель: Проверить практическое качество генерируемого текста")
    print("📊 Модели: Лучшие результаты 1K экспериментов")
    
    # Проверяем наличие sample_centered.py
    if not os.path.exists('sample_centered.py'):
        print("⚠️  sample_centered.py не найден, создаем...")
        # Можно создать или скопировать из sample.py с изменениями
        print("ℹ️  Используем стандартный sample.py для всех моделей")
    
    # Запускаем сравнение
    all_results = compare_generations()
    
    if all_results:
        # Анализируем качество
        quality_metrics = analyze_generation_quality(all_results)
        
        print(f"\n🎉 Тестирование завершено!")
        print(f"📊 Протестировано моделей: {len(all_results)}")
        print(f"💾 Результаты сохранены в training_plots/")
    else:
        print("❌ Не удалось протестировать ни одну модель")

if __name__ == "__main__":
    main()
