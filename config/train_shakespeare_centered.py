# Эксперимент с центрированием векторов в nanoGPT
# Тестируем разные варианты центрирования на датасете Shakespeare

out_dir = 'out-shakespeare-centered'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-centered'
wandb_run_name = 'centered-experiment'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Модель с центрированием
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Параметры центрирования
use_centered_attention = True     # Использовать центрированный attention
center_qk = True                  # Центрировать query/key векторы
center_block_output = False       # Центрировать выход каждого блока
center_final_output = True        # Центрировать финальные эмбеддинги
centering_mode = 'adaptive'       # 'simple', 'adaptive', 'learnable_center', 'momentum'

# Обучение
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
