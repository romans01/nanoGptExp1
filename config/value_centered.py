
# Базовая конфигурация для экспериментов с расширенным центрированием
out_dir = 'out-advanced-centering-value_centered'
eval_interval = 50
log_interval = 10
eval_iters = 20
eval_only = False
always_save_checkpoint = True

# Данные
dataset = 'shakespeare'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 256

# Модель - уменьшенная для быстрых экспериментов
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.1

# Оптимизация
learning_rate = 3e-4
max_iters = 150
lr_decay_iters = 150
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 100
weight_decay = 1e-1

# Система
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Центрирование
center_v = True
centering_mode = 'adaptive' 
