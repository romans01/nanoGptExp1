#!/usr/bin/env python3
"""
Тестирование качества генерации для финальных результатов
"""

import os
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Any

# Финальные модели для тестирования (по результатам 1000 итераций)
final_models = [
    {
        'name': 'aggressive_all_1000',
        'description': '🥇 ПОБЕДИТЕЛЬ: Aggressive All (QK+V+Emb+MLP)',
        'out_dir': 'out-final-aggressive_all_1000',
        'val_loss': 4.8281,
        'script': 'sample_centered.py'
    },
    {
        'name': 'value_centered_1000', 
        'description': '🥈 2-е место: Value центрирование',
        'out_dir': 'out-final-value_centered_1000',
        'val_loss': 4.9422,
        'script': 'sample_centered.py'
    },
    {
        'name': 'baseline_1000',
        'description': '🥉 3-е место: Baseline (без центрирования)',
        'out_dir': 'out-final-baseline_1000', 
        'val_loss': 4.9616,
        'script': 'sample.py'
    }
]

# Промпты для тестирования разных аспектов
test_prompts = [
    {
        'name': 'shakespeare_dialogue',
        'prompt': 'JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?',
        'description': 'Шекспировский диалог'
    },
    {
        'name': 'dramatic_monologue',
        'prompt': 'HAMLET:\nTo be, or not to be, that is the question:',
        'description': 'Драматический монолог'
    },
    {
        'name': 'character_introduction',
        'prompt': 'Enter KING RICHARD II, with his nobles.',
        'description': 'Введение персонажа'
    },
    {
        'name': 'poetic_verse',
        'prompt': 'Shall I compare thee to a summer\'s day?',
        'description': 'Поэтические строки'
    },
    {
        'name': 'simple_start',
        'prompt': 'The king said',
        'description': 'Простое начало'
    }
]

def generate_text(model_info: Dict[str, Any], prompt: str, num_tokens: int = 500) -> Dict[str, Any]:
    """Генерирует текст для модели"""
    
    print(f"🎭 Генерация для {model_info['name']}...")
    print(f"   📝 Промпт: \"{prompt[:50]}...\"")
    
    try:
        # Команда для генерации
        cmd = [
            'python', model_info['script'],
            f'--out_dir={model_info["out_dir"]}',
            f'--start={prompt}',
            '--num_samples=1',
            f'--max_new_tokens={num_tokens}',
            '--temperature=0.8',
            '--top_k=200',
            '--seed=42'  # Фиксированный seed для воспроизводимости
        ]
        
        start_time = time.time()
        
        # Запускаем генерацию
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 минута таймаут
        )
        
        generation_time = time.time() - start_time
        
        if result.returncode == 0:
            # Извлекаем сгенерированный текст
            output_lines = result.stdout.split('\n')
            
            # Ищем начало генерации (после информации о модели)
            generation_start = -1
            for i, line in enumerate(output_lines):
                if prompt in line:
                    generation_start = i
                    break
            
            if generation_start >= 0:
                # Собираем сгенерированный текст
                generated_lines = []
                for line in output_lines[generation_start:]:
                    if line.strip() == '---------------':
                        break
                    generated_lines.append(line)
                
                generated_text = '\n'.join(generated_lines).strip()
                
                return {
                    'success': True,
                    'generated_text': generated_text,
                    'generation_time': generation_time,
                    'model_info': model_info,
                    'prompt': prompt
                }
            else:
                return {
                    'success': False,
                    'error': 'Could not find generated text in output',
                    'raw_output': result.stdout[:500],
                    'model_info': model_info,
                    'prompt': prompt
                }
        else:
            return {
                'success': False,
                'error': f'Generation failed: {result.stderr[:200]}',
                'model_info': model_info,
                'prompt': prompt
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Generation timeout (60s)',
            'model_info': model_info,
            'prompt': prompt
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Exception: {str(e)}',
            'model_info': model_info,
            'prompt': prompt
        }

def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Анализирует качество сгенерированного текста"""
    
    lines = text.split('\n')
    words = text.split()
    
    # Базовые метрики
    metrics = {
        'total_chars': len(text),
        'total_words': len(words),
        'total_lines': len(lines),
        'avg_words_per_line': len(words) / max(len(lines), 1),
        'avg_chars_per_word': len(text.replace(' ', '')) / max(len(words), 1)
    }
    
    # Анализ структуры Шекспира
    shakespeare_features = {
        'has_character_names': sum(1 for line in lines if ':' in line and line.isupper()),
        'has_stage_directions': sum(1 for line in lines if line.strip().startswith('[')),
        'has_dialogue_structure': sum(1 for line in lines if ':' in line),
        'blank_lines': sum(1 for line in lines if not line.strip()),
    }
    
    # Оценка повторяемости
    word_counts = {}
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    if words:
        unique_words = len(word_counts)
        repetition_score = unique_words / len(words)  # Чем выше, тем меньше повторов
    else:
        repetition_score = 0
    
    # Оценка связности (простая)
    coherence_indicators = [
        'and', 'but', 'then', 'so', 'because', 'therefore', 'however', 'thus',
        'when', 'where', 'while', 'since', 'although', 'though'
    ]
    
    coherence_words = sum(1 for word in words if word.lower().strip('.,!?;:') in coherence_indicators)
    coherence_score = coherence_words / max(len(words), 1)
    
    return {
        'basic_metrics': metrics,
        'shakespeare_features': shakespeare_features,
        'repetition_score': repetition_score,
        'coherence_score': coherence_score,
        'unique_words': unique_words if 'unique_words' in locals() else 0
    }

def run_comprehensive_generation_test():
    """Запускает комплексное тестирование генерации"""
    
    print("🎭 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ КАЧЕСТВА ГЕНЕРАЦИИ")
    print("=" * 80)
    print("🎯 Цель: Сравнить качество генерации финальных моделей")
    print("📊 Модели: 3 финальные модели")
    print("📝 Промпты: 5 различных типов")
    print("🔤 Токены: 500 на каждую генерацию")
    print("=" * 80)
    
    all_results = []
    
    # Проверяем наличие моделей
    print("\n🔍 ПРОВЕРКА ДОСТУПНОСТИ МОДЕЛЕЙ:")
    for model in final_models:
        ckpt_path = os.path.join(model['out_dir'], 'ckpt.pt')
        if os.path.exists(ckpt_path):
            print(f"   ✅ {model['name']}: модель найдена")
        else:
            print(f"   ❌ {model['name']}: модель НЕ найдена в {ckpt_path}")
    
    print(f"\n🚀 НАЧИНАЕМ ГЕНЕРАЦИЮ...")
    
    # Тестируем каждую комбинацию модель-промпт
    total_tests = len(final_models) * len(test_prompts)
    current_test = 0
    
    for model in final_models:
        print(f"\n📱 МОДЕЛЬ: {model['description']}")
        print(f"   📊 Validation Loss: {model['val_loss']}")
        print("-" * 60)
        
        model_results = []
        
        for prompt_info in test_prompts:
            current_test += 1
            print(f"\n   🎭 Тест {current_test}/{total_tests}: {prompt_info['description']}")
            
            # Генерируем текст
            result = generate_text(model, prompt_info['prompt'], 500)
            
            if result['success']:
                # Анализируем качество
                quality_analysis = analyze_text_quality(result['generated_text'])
                result['quality_analysis'] = quality_analysis
                
                print(f"      ✅ Успешно ({result['generation_time']:.1f}с)")
                print(f"      📊 Слов: {quality_analysis['basic_metrics']['total_words']}")
                print(f"      🔄 Уникальность: {quality_analysis['repetition_score']:.3f}")
                
            else:
                print(f"      ❌ Ошибка: {result['error']}")
            
            result['prompt_info'] = prompt_info
            model_results.append(result)
            
            # Небольшая пауза между генерациями
            time.sleep(1)
        
        all_results.extend(model_results)
    
    return all_results

def create_generation_report(results: List[Dict[str, Any]]):
    """Создает отчет по результатам генерации"""
    
    print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ ГЕНЕРАЦИИ")
    print("=" * 60)
    
    # Группируем по моделям
    model_results = {}
    for result in results:
        model_name = result['model_info']['name']
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)
    
    # Анализируем каждую модель
    model_scores = {}
    
    for model_name, model_tests in model_results.items():
        print(f"\n🎭 {model_name.upper()}:")
        print("-" * 40)
        
        successful_tests = [r for r in model_tests if r['success']]
        failed_tests = [r for r in model_tests if not r['success']]
        
        print(f"   ✅ Успешных генераций: {len(successful_tests)}/{len(model_tests)}")
        
        if failed_tests:
            print(f"   ❌ Неудачных: {len(failed_tests)}")
            for failed in failed_tests:
                print(f"      • {failed['prompt_info']['description']}: {failed['error']}")
        
        if successful_tests:
            # Агрегированные метрики
            avg_words = sum(r['quality_analysis']['basic_metrics']['total_words'] for r in successful_tests) / len(successful_tests)
            avg_repetition = sum(r['quality_analysis']['repetition_score'] for r in successful_tests) / len(successful_tests)
            avg_coherence = sum(r['quality_analysis']['coherence_score'] for r in successful_tests) / len(successful_tests)
            avg_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
            
            # Шекспировские особенности
            avg_character_names = sum(r['quality_analysis']['shakespeare_features']['has_character_names'] for r in successful_tests) / len(successful_tests)
            avg_dialogue = sum(r['quality_analysis']['shakespeare_features']['has_dialogue_structure'] for r in successful_tests) / len(successful_tests)
            
            print(f"   📊 Средние метрики:")
            print(f"      🔤 Слов: {avg_words:.1f}")
            print(f"      🔄 Уникальность: {avg_repetition:.3f}")
            print(f"      🔗 Связность: {avg_coherence:.3f}")
            print(f"      ⏱️  Время: {avg_time:.1f}с")
            print(f"      🎭 Персонажи: {avg_character_names:.1f}")
            print(f"      💬 Диалоги: {avg_dialogue:.1f}")
            
            # Общий балл (простая формула)
            quality_score = (
                avg_repetition * 0.3 +  # Уникальность важна
                avg_coherence * 0.2 +   # Связность
                min(avg_words / 100, 1.0) * 0.2 +  # Длина (до 100 слов = хорошо)
                (avg_character_names > 0) * 0.15 +  # Есть персонажи
                (avg_dialogue > 0) * 0.15           # Есть диалоги
            )
            
            model_scores[model_name] = {
                'quality_score': quality_score,
                'avg_words': avg_words,
                'avg_repetition': avg_repetition,
                'avg_coherence': avg_coherence,
                'avg_time': avg_time,
                'success_rate': len(successful_tests) / len(model_tests)
            }
            
            print(f"      🏆 Общий балл качества: {quality_score:.3f}")
    
    # Финальный рейтинг
    print(f"\n🏆 ФИНАЛЬНЫЙ РЕЙТИНГ ПО КАЧЕСТВУ ГЕНЕРАЦИИ:")
    print("=" * 50)
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['quality_score'], reverse=True)
    
    for i, (model_name, scores) in enumerate(sorted_models, 1):
        model_info = next(m for m in final_models if m['name'] == model_name)
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        
        print(f"{medal} {model_name}:")
        print(f"   📝 {model_info['description']}")
        print(f"   🏆 Качество: {scores['quality_score']:.3f}")
        print(f"   📊 Val Loss: {model_info['val_loss']}")
        print(f"   ✅ Успешность: {scores['success_rate']*100:.0f}%")
        print()
    
    return model_scores

def save_generation_samples(results: List[Dict[str, Any]]):
    """Сохраняет образцы генерации"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем папку для образцов
    samples_dir = f'generation_samples_{timestamp}'
    os.makedirs(samples_dir, exist_ok=True)
    
    print(f"\n💾 СОХРАНЕНИЕ ОБРАЗЦОВ ГЕНЕРАЦИИ:")
    print(f"📁 Папка: {samples_dir}")
    
    for result in results:
        if result['success']:
            model_name = result['model_info']['name']
            prompt_name = result['prompt_info']['name']
            
            filename = f"{model_name}_{prompt_name}.txt"
            filepath = os.path.join(samples_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"МОДЕЛЬ: {result['model_info']['description']}\n")
                f.write(f"VALIDATION LOSS: {result['model_info']['val_loss']}\n")
                f.write(f"ПРОМПТ: {result['prompt_info']['description']}\n")
                f.write(f"ВРЕМЯ ГЕНЕРАЦИИ: {result['generation_time']:.1f}с\n")
                f.write("=" * 60 + "\n\n")
                f.write(result['generated_text'])
                f.write("\n\n" + "=" * 60 + "\n")
                
                # Добавляем метрики качества
                qa = result['quality_analysis']
                f.write("МЕТРИКИ КАЧЕСТВА:\n")
                f.write(f"• Слов: {qa['basic_metrics']['total_words']}\n")
                f.write(f"• Уникальность: {qa['repetition_score']:.3f}\n")
                f.write(f"• Связность: {qa['coherence_score']:.3f}\n")
                f.write(f"• Персонажи: {qa['shakespeare_features']['has_character_names']}\n")
                f.write(f"• Диалоги: {qa['shakespeare_features']['has_dialogue_structure']}\n")
            
            print(f"   💾 {filename}")
    
    print(f"✅ Сохранено образцов: {len([r for r in results if r['success']])}")

def main():
    """Главная функция"""
    
    print("🎭 ТЕСТИРОВАНИЕ КАЧЕСТВА ГЕНЕРАЦИИ ФИНАЛЬНЫХ МОДЕЛЕЙ")
    print("=" * 80)
    
    # Запускаем тестирование
    results = run_comprehensive_generation_test()
    
    # Анализируем результаты
    model_scores = create_generation_report(results)
    
    # Сохраняем образцы
    save_generation_samples(results)
    
    print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("📊 Результаты показывают качество генерации каждой модели")
    print("💾 Все образцы сохранены для детального анализа")
    
    return results, model_scores

if __name__ == "__main__":
    main()
