
# Конфигурация для финального тестирования лучших моделей расширенного центрирования
out_dir = 'out-advanced-1k-embeddings_centered_1k'
eval_interval = 100
log_interval = 20
eval_iters = 50
eval_only = False
always_save_checkpoint = True

# Данные
dataset = 'shakespeare'
gradient_accumulation_steps = 4
batch_size = 8
block_size = 256

# Модель - полноразмерная для серьезного тестирования
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# Оптимизация
learning_rate = 3e-4
max_iters = 1000
lr_decay_iters = 1000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200
weight_decay = 1e-1

# Система
device = 'cuda'
dtype = 'bfloat16'
compile = True

# Центрирование
center_embeddings = True
centering_mode = 'adaptive' 
