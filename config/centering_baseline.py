# Эксперимент: baseline
# Базовая модель без центрирования

# Базовые настройки
out_dir = 'out-centering-baseline'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'centering-experiments'
wandb_run_name = 'baseline'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# Архитектура модели
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Параметры обучения
learning_rate = 1e-3
max_iters = 1000  # Короткие эксперименты
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Параметры центрирования
use_centered_attention = False
center_qk = False
center_block_output = False
center_final_output = False