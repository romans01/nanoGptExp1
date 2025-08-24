#!/usr/bin/env python3
"""
Комплексное тестирование генерации для лучших моделей с центрированием
Генерация 500 токенов для качественного анализа
"""

import os
import subprocess
import time
from datetime import datetime

def test_model_generation(model_path, model_name, script_name, prompt, max_tokens=500):
    """Тестирует генерацию для одной модели"""
    
    print(f"\n🎭 {model_name}")
    print("=" * 60)
    print(f"📁 Модель: {model_path}")
    print(f"🎯 Промпт: '{prompt}'")
    print(f"📊 Токенов: {max_tokens}")
    print("-" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return None
    
    try:
        cmd = [
            'python', script_name,
            '--out_dir', model_path,
            '--start', prompt,
            '--num_samples', '1',
            '--max_new_tokens', str(max_tokens),
            '--temperature', '0.8',
            '--top_k', '200'
        ]
        
        print(f"🔄 Запуск генерации...")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        generation_time = time.time() - start_time
        
        if result.returncode == 0:
            # Извлекаем сгенерированный текст
            output_lines = result.stdout.strip().split('\n')
            
            # Ищем начало генерации
            generation_started = False
            generated_text = []
            
            for line in output_lines:
                if prompt in line and not generation_started:
                    generation_started = True
                    generated_text.append(line)
                elif generation_started and not line.startswith('---------------'):
                    generated_text.append(line)
                elif line.startswith('---------------'):
                    break
            
            if generated_text:
                full_text = '\n'.join(generated_text)
                print(f"✅ Генерация завершена за {generation_time:.1f}с")
                print(f"📝 Длина: {len(full_text)} символов")
                print()
                print(full_text)
                print()
                return {
                    'text': full_text,
                    'time': generation_time,
                    'length': len(full_text),
                    'success': True
                }
            else:
                print("⚠️  Не удалось извлечь сгенерированный текст")
                return {'success': False, 'error': 'Extraction failed'}
        else:
            print(f"❌ Ошибка генерации: {result.stderr[:200]}")
            return {'success': False, 'error': result.stderr[:200]}
            
    except subprocess.TimeoutExpired:
        print("⏱️  Таймаут генерации (120с)")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return {'success': False, 'error': str(e)}

def comprehensive_comparison():
    """Проводит комплексное сравнение лучших моделей"""
    
    print("🎭 КОМПЛЕКСНОЕ СРАВНЕНИЕ ГЕНЕРАЦИИ (500 ТОКЕНОВ)")
    print("=" * 80)
    
    # Лучшие модели по результатам 1K экспериментов
    models = [
        {
            'path': 'out-advanced-1k-baseline_1k',
            'name': '📊 BASELINE 1K',
            'script': 'sample.py',
            'description': 'Контрольная группа (без центрирования)'
        },
        {
            'path': 'out-advanced-1k-embeddings_centered_1k', 
            'name': '🥇 EMBEDDINGS CENTERED 1K',
            'script': 'sample_centered.py',
            'description': 'ПОБЕДИТЕЛЬ: +2.26% улучшение'
        },
        {
            'path': 'out-advanced-1k-qk_plus_value_1k',
            'name': '🥈 QK + VALUE 1K',
            'script': 'sample_centered.py',
            'description': '2-е место: +1.89% улучшение'
        },
        {
            'path': 'out-advanced-1k-value_centered_1k',
            'name': '🥉 VALUE CENTERED 1K',
            'script': 'sample_centered.py',
            'description': '3-е место: +1.13% улучшение'
        }
    ]
    
    # Разнообразные промпты для тестирования
    prompts = [
        "ROMEO:",
        "JULIET:",
        "To be or not to be",
        "Once upon a time in a kingdom far away",
        "The meaning of life is"
    ]
    
    all_results = {}
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"🎯 ТЕСТИРОВАНИЕ ПРОМПТА: '{prompt}'")
        print(f"{'='*80}")
        
        prompt_results = {}
        
        for model in models:
            result = test_model_generation(
                model['path'],
                model['name'],
                model['script'],
                prompt,
                max_tokens=500
            )
            
            if result:
                prompt_results[model['name']] = {
                    'result': result,
                    'description': model['description']
                }
        
        all_results[prompt] = prompt_results
        
        # Небольшая пауза между промптами
        time.sleep(2)
    
    # Создаем детальный отчет
    create_detailed_report(all_results)
    
    return all_results

def create_detailed_report(all_results):
    """Создает детальный отчет сравнения"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"training_plots/comprehensive_generation_report_{timestamp}.txt"
    
    os.makedirs('training_plots', exist_ok=True)
    
    print(f"\n📊 СОЗДАНИЕ ДЕТАЛЬНОГО ОТЧЕТА...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🎭 КОМПЛЕКСНОЕ СРАВНЕНИЕ ГЕНЕРАЦИИ (500 ТОКЕНОВ)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ:\n")
        f.write("🥇 Embeddings Centered: +2.26% улучшение validation loss\n")
        f.write("🥈 QK + Value: +1.89% улучшение validation loss\n") 
        f.write("🥉 Value Centered: +1.13% улучшение validation loss\n")
        f.write("📊 Baseline: контрольная группа\n\n")
        
        f.write("🎯 ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:\n")
        f.write("• Токенов: 500\n")
        f.write("• Температура: 0.8\n")
        f.write("• Top-K: 200\n")
        f.write("• Модели: обучены на 1000 итераций\n\n")
        
        # Статистика по каждому промпту
        for prompt, prompt_results in all_results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"🎯 ПРОМПТ: '{prompt}'\n")
            f.write(f"{'='*80}\n\n")
            
            # Сначала статистика
            f.write("📊 СТАТИСТИКА ГЕНЕРАЦИИ:\n")
            f.write("-" * 40 + "\n")
            
            for model_name, data in prompt_results.items():
                if data['result']['success']:
                    result = data['result']
                    f.write(f"{model_name}:\n")
                    f.write(f"  📝 Длина: {result['length']} символов\n")
                    f.write(f"  ⏱️  Время: {result['time']:.1f}с\n")
                    f.write(f"  📋 Описание: {data['description']}\n\n")
            
            # Затем полные тексты
            f.write("\n📝 СГЕНЕРИРОВАННЫЕ ТЕКСТЫ:\n")
            f.write("-" * 40 + "\n\n")
            
            for model_name, data in prompt_results.items():
                if data['result']['success']:
                    f.write(f"{model_name}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{data['result']['text']}\n\n")
                    f.write("=" * 60 + "\n\n")
        
        # Общая статистика
        f.write(f"\n📊 ОБЩАЯ СТАТИСТИКА:\n")
        f.write("=" * 40 + "\n")
        
        total_stats = {}
        for prompt, prompt_results in all_results.items():
            for model_name, data in prompt_results.items():
                if data['result']['success']:
                    if model_name not in total_stats:
                        total_stats[model_name] = {
                            'total_length': 0,
                            'total_time': 0,
                            'count': 0
                        }
                    
                    total_stats[model_name]['total_length'] += data['result']['length']
                    total_stats[model_name]['total_time'] += data['result']['time']
                    total_stats[model_name]['count'] += 1
        
        for model_name, stats in total_stats.items():
            if stats['count'] > 0:
                avg_length = stats['total_length'] / stats['count']
                avg_time = stats['total_time'] / stats['count']
                f.write(f"\n{model_name}:\n")
                f.write(f"  📊 Тестов: {stats['count']}\n")
                f.write(f"  📝 Средняя длина: {avg_length:.0f} символов\n")
                f.write(f"  ⏱️  Среднее время: {avg_time:.1f}с\n")
        
        f.write(f"\n\n📅 Отчет создан: {timestamp}\n")
        f.write("🎯 Все модели обучены на 1000 итераций Shakespeare BPE\n")
    
    print(f"💾 Детальный отчет сохранен: {report_path}")
    
    # Также выводим краткую статистику в консоль
    print(f"\n📊 КРАТКАЯ СТАТИСТИКА:")
    print("-" * 40)
    
    for model_name, stats in total_stats.items():
        if stats['count'] > 0:
            avg_length = stats['total_length'] / stats['count']
            avg_time = stats['total_time'] / stats['count']
            print(f"{model_name}:")
            print(f"  📊 Успешных тестов: {stats['count']}")
            print(f"  📝 Средняя длина: {avg_length:.0f} символов")
            print(f"  ⏱️  Среднее время: {avg_time:.1f}с")
            print()

def analyze_quality_metrics(all_results):
    """Анализирует качественные метрики генераций"""
    
    print(f"\n🔍 АНАЛИЗ КАЧЕСТВЕННЫХ МЕТРИК")
    print("=" * 50)
    
    quality_analysis = {}
    
    for prompt, prompt_results in all_results.items():
        print(f"\n🎯 Промпт: '{prompt}'")
        print("-" * 30)
        
        for model_name, data in prompt_results.items():
            if data['result']['success']:
                text = data['result']['text']
                
                # Простые метрики качества
                words = text.split()
                unique_words = set(words)
                
                # Шекспировские слова
                shakespeare_words = ['thou', 'thee', 'thy', 'hath', 'doth', 'art', 'shall', 
                                   'whence', 'wherefore', 'methinks', 'prithee', 'ere']
                shakespeare_count = sum(1 for word in words if word.lower() in shakespeare_words)
                
                # Персонажи
                characters = ['ROMEO', 'JULIET', 'MERCUTIO', 'BENVOLIO', 'NURSE', 'FRIAR', 'CAPULET']
                character_count = sum(1 for char in characters if char in text)
                
                metrics = {
                    'word_count': len(words),
                    'unique_ratio': len(unique_words) / len(words) if words else 0,
                    'shakespeare_density': shakespeare_count / len(words) if words else 0,
                    'character_diversity': character_count
                }
                
                if model_name not in quality_analysis:
                    quality_analysis[model_name] = []
                quality_analysis[model_name].append(metrics)
                
                print(f"  {model_name}:")
                print(f"    📝 Слов: {metrics['word_count']}")
                print(f"    🔄 Уникальность: {metrics['unique_ratio']:.2f}")
                print(f"    🎭 Шекспир-стиль: {metrics['shakespeare_density']:.3f}")
                print(f"    👥 Персонажей: {metrics['character_diversity']}")
    
    # Средние метрики
    print(f"\n📊 СРЕДНИЕ МЕТРИКИ ПО МОДЕЛЯМ:")
    print("=" * 40)
    
    for model_name, metrics_list in quality_analysis.items():
        if metrics_list:
            avg_metrics = {
                'word_count': sum(m['word_count'] for m in metrics_list) / len(metrics_list),
                'unique_ratio': sum(m['unique_ratio'] for m in metrics_list) / len(metrics_list),
                'shakespeare_density': sum(m['shakespeare_density'] for m in metrics_list) / len(metrics_list),
                'character_diversity': sum(m['character_diversity'] for m in metrics_list) / len(metrics_list)
            }
            
            print(f"\n{model_name}:")
            print(f"  📝 Среднее слов: {avg_metrics['word_count']:.0f}")
            print(f"  🔄 Средняя уникальность: {avg_metrics['unique_ratio']:.3f}")
            print(f"  🎭 Средний Шекспир-стиль: {avg_metrics['shakespeare_density']:.4f}")
            print(f"  👥 Средние персонажи: {avg_metrics['character_diversity']:.1f}")

def main():
    print("🎭 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ ГЕНЕРАЦИИ")
    print("🎯 Цель: Глубокий анализ качества генерации (500 токенов)")
    print("📊 Модели: Лучшие результаты 1K экспериментов")
    print("⏱️  Ожидаемое время: ~10-15 минут")
    
    start_time = time.time()
    
    # Проверяем наличие моделей
    required_models = [
        'out-advanced-1k-baseline_1k',
        'out-advanced-1k-embeddings_centered_1k',
        'out-advanced-1k-qk_plus_value_1k',
        'out-advanced-1k-value_centered_1k'
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    if missing_models:
        print(f"⚠️  Отсутствующие модели: {missing_models}")
        print("🔄 Убедитесь, что все модели обучены")
    
    # Запускаем комплексное сравнение
    all_results = comprehensive_comparison()
    
    if all_results:
        # Анализируем качественные метрики
        analyze_quality_metrics(all_results)
        
        total_time = time.time() - start_time
        print(f"\n🎉 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print(f"⏱️  Общее время: {total_time:.1f}с ({total_time/60:.1f} мин)")
        print(f"📊 Протестировано промптов: {len(all_results)}")
        print(f"💾 Результаты сохранены в training_plots/")
    else:
        print("❌ Тестирование не удалось")

if __name__ == "__main__":
    main()
