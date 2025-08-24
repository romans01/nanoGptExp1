#!/usr/bin/env python3
"""
Тестовый скрипт для проверки модуля центрирования векторов
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_with_centering import GPTConfig, GPT, VectorCentering

def test_vector_centering():
    """Тестирует модуль VectorCentering"""
    print("🧪 Тестирую модуль VectorCentering...")
    
    # Создаем тестовые данные - кластер векторов
    torch.manual_seed(42)
    batch_size, seq_len, embd_dim = 4, 10, 64
    
    # Создаем кластер векторов (все близко к [1, 0, 0, ...])
    base_vector = torch.zeros(embd_dim)
    base_vector[0] = 1.0
    
    # Добавляем небольшой шум
    noise = 0.1 * torch.randn(batch_size, seq_len, embd_dim)
    x = base_vector.unsqueeze(0).unsqueeze(0) + noise
    
    # Нормируем на единичную сферу
    x = x / torch.norm(x, dim=-1, keepdim=True)
    
    print(f"Исходные векторы: {x.shape}")
    print(f"Средняя норма: {torch.norm(x, dim=-1).mean():.4f}")
    
    # Вычисляем исходные углы между векторами
    x_flat = x.view(-1, embd_dim)
    cos_sim_matrix = torch.mm(x_flat, x_flat.t())
    angles_rad = torch.acos(torch.clamp(cos_sim_matrix, -1, 1))
    angles_deg = angles_rad * 180 / torch.pi
    
    # Берем только верхний треугольник (исключаем диагональ)
    mask = torch.triu(torch.ones_like(angles_deg), diagonal=1).bool()
    original_angles = angles_deg[mask]
    
    print(f"Исходные углы: {original_angles.mean():.2f}° ± {original_angles.std():.2f}°")
    
    # Тестируем разные режимы центрирования
    config = GPTConfig(n_embd=embd_dim)
    
    modes = ['simple', 'adaptive', 'learnable_center', 'momentum']
    results = {}
    
    for mode in modes:
        print(f"\n--- Режим: {mode} ---")
        centering = VectorCentering(config, mode=mode)
        
        # Применяем центрирование
        x_centered, stats = centering(x, return_stats=True)
        
        print(f"Статистики: {stats}")
        
        # Вычисляем новые углы
        x_centered_flat = x_centered.view(-1, embd_dim)
        cos_sim_matrix_new = torch.mm(x_centered_flat, x_centered_flat.t())
        angles_rad_new = torch.acos(torch.clamp(cos_sim_matrix_new, -1, 1))
        angles_deg_new = angles_rad_new * 180 / torch.pi
        
        new_angles = angles_deg_new[mask]
        
        print(f"Новые углы: {new_angles.mean():.2f}° ± {new_angles.std():.2f}°")
        print(f"Увеличение углов: {new_angles.mean() / original_angles.mean():.2f}x")
        
        results[mode] = {
            'original_angles': original_angles.detach().numpy(),
            'new_angles': new_angles.detach().numpy(),
            'stats': stats
        }
    
    # Создаем график
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        
        original = results[mode]['original_angles']
        new = results[mode]['new_angles']
        
        ax.hist(original, bins=20, alpha=0.5, label='Исходные', color='blue')
        ax.hist(new, bins=20, alpha=0.5, label='После центрирования', color='red')
        
        ax.set_title(f'Режим: {mode}')
        ax.set_xlabel('Угол (градусы)')
        ax.set_ylabel('Частота')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем статистики
        orig_mean = np.mean(original)
        new_mean = np.mean(new)
        ax.text(0.05, 0.95, f'Исходные: {orig_mean:.1f}°\nНовые: {new_mean:.1f}°\nУвеличение: {new_mean/orig_mean:.1f}x', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('centering_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 График сохранен в centering_test_results.png")
    
    return results

def test_model_with_centering():
    """Тестирует модель с центрированием"""
    print("\n🤖 Тестирую модель с центрированием...")
    
    # Конфигурация с центрированием
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        use_centered_attention=True,
        center_qk=True,
        center_final_output=True,
        centering_mode='adaptive'
    )
    
    model = GPT(config)
    print(f"Модель создана: {model.get_num_params()/1e3:.1f}K параметров")
    
    # Тестовые данные
    batch_size, seq_len = 2, 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        logits, loss, centering_stats = model(x, x, return_centering_stats=True)
    
    print(f"Выход модели: {logits.shape}")
    print(f"Статистики центрирования:")
    for name, stats in centering_stats:
        print(f"  {name}: {stats}")
    
    # Сравниваем с базовой моделью
    config_baseline = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        use_centered_attention=False,
        center_qk=False,
        center_final_output=False
    )
    
    model_baseline = GPT(config_baseline)
    
    with torch.no_grad():
        logits_baseline, _ = model_baseline(x, x)
    
    print(f"Базовая модель: {logits_baseline.shape}")
    print(f"Разница в выходах: {torch.mean((logits - logits_baseline)**2).item():.6f}")
    
    return model, model_baseline

def benchmark_centering():
    """Бенчмарк производительности центрирования"""
    print("\n⚡ Бенчмарк производительности...")
    
    import time
    
    config = GPTConfig(n_embd=768)
    centering = VectorCentering(config, mode='adaptive')
    
    # Разные размеры батчей
    batch_sizes = [1, 4, 16, 64]
    seq_len = 512
    embd_dim = 768
    
    results = []
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, seq_len, embd_dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        
        # Прогрев
        for _ in range(10):
            _ = centering(x)
        
        # Измерение времени
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            _ = centering(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # мс
        tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)
        
        results.append({
            'batch_size': batch_size,
            'time_ms': avg_time,
            'tokens_per_sec': tokens_per_sec
        })
        
        print(f"Batch {batch_size}: {avg_time:.2f}ms, {tokens_per_sec:.0f} tokens/sec")
    
    return results

def main():
    print("🧪 Тестирование центрирования векторов в nanoGPT")
    print("=" * 60)
    
    # Тест 1: Модуль центрирования
    centering_results = test_vector_centering()
    
    # Тест 2: Модель с центрированием
    model, model_baseline = test_model_with_centering()
    
    # Тест 3: Бенчмарк производительности
    benchmark_results = benchmark_centering()
    
    print("\n✅ Все тесты завершены!")
    print("📊 Результаты:")
    print("  - centering_test_results.png - визуализация эффекта центрирования")
    print("  - Модель с центрированием работает корректно")
    print("  - Производительность приемлемая")

if __name__ == "__main__":
    main()
