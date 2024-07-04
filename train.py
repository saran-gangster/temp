import os
import re
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
from tqdm import tqdm
import pickle
from functools import partial
from src.model import RWKV, RWKVConfig, create_model
from src.tokenizer import RWKVTokenizer
from src.binidx import MMapIndexedDataset
jax.config.update("jax_log_compiles", True)
jax.config.update("jax_debug_nans", True)


MODEL_PATH = './weight/RWKV-x060.pkl'
TOKENIZER_PATH = "rwkv_vocab_v20230424.txt"
DATA_PATH = 'data/minipile'
SAVE_PATH = os.path.abspath("rwk34612v")

config = RWKVConfig(
    vocab_size=50277,
    n_layer=2,
    n_embd=128,
    dim_att=128,
    dim_ffn=512,
    head_size_a=32,
    n_head=4,
    head_size_divisor=8,
    dropout=0.1,
    layer_norm_epsilon=1e-5,
    chunk_size=32,
    subchunk_size=64,
    min_clamp = 0.05
)

INITIAL_LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 1e-3
WARMUP_STEPS = 1000
DECAY_STEPS = 10000
BATCH_SIZE = 32
SEQ_LEN = 512
EPOCHS = 2
SAVE_EVERY = 1000
GRAD_CLIP_NORM = 0.5  
GRAD_CLIP_VALUE = 0.5  
EPSILON = 1e-8
LABEL_SMOOTHING = 0.1
global_step = 0

devices = jax.local_devices()
num_devices = len(devices)
print(f"Number of devices: {num_devices}")

BATCH_SIZE_PER_DEVICE = BATCH_SIZE // num_devices
assert BATCH_SIZE % num_devices == 0, f"Batch size must be divisible by the number of devices. Got {BATCH_SIZE} and {num_devices} devices."

tokenizer = RWKVTokenizer(TOKENIZER_PATH)

def init_or_load_model(config, model_path):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        model, _ = create_model(config)  # Create model to get summary
    else:
        print("Creating new model")
        model, params = create_model(config)
        with open(model_path, 'wb') as f:
            pickle.dump(params, f)
    return model, params

model, params = init_or_load_model(config, MODEL_PATH)

def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Checkpoint loaded, step: {checkpoint['step']}")
    return checkpoint

def save_checkpoint(train_state, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = os.path.join(save_dir, "checkpoint")
    
    raw_params = jax.device_get(train_state.params)

    checkpoint = {
        'params': raw_params,
        'step': step
    }

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved at step {step}")

def create_learning_rate_schedule():
    def schedule(step):
        warmup_factor = jnp.minimum(step / WARMUP_STEPS, 1.0)
        warmup_lr = INITIAL_LEARNING_RATE + (MAX_LEARNING_RATE - INITIAL_LEARNING_RATE) * warmup_factor
        
        decay_factor = jnp.maximum(0.0, 1.0 - (step - WARMUP_STEPS) / DECAY_STEPS)
        decay_lr = MAX_LEARNING_RATE * decay_factor
        
        return jnp.where(step < WARMUP_STEPS, warmup_lr, decay_lr)
    return schedule

def create_train_state(params, learning_rate_schedule):
    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.clip(GRAD_CLIP_VALUE),
        optax.adamw(learning_rate_schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint_')]
    if not checkpoints:
        return None
    
    def extract_step(checkpoint_name):
        match = re.search(r'checkpoint_(\d+)', checkpoint_name)
        return int(match.group(1)) if match else -1
    
    latest_checkpoint = max(checkpoints, key=extract_step)
    return os.path.join(checkpoint_dir, latest_checkpoint)

dataset = MMapIndexedDataset(DATA_PATH)

def pad_sequences(sequences, max_len, pad_value=0):
    padded = []
    for seq in sequences:
        if isinstance(seq, (list, np.ndarray)):
            flat_seq = np.concatenate(seq) if isinstance(seq, list) else seq
        else:
            flat_seq = np.array(seq)
        
        if len(flat_seq) > max_len:
            padded.append(flat_seq[:max_len])
        else:
            padded.append(np.pad(flat_seq, (0, max_len - len(flat_seq)), constant_values=pad_value))
    return np.array(padded, dtype=np.int32)

def create_mask(padded_sequences, max_len):
    return np.array([[1 if i < np.sum(seq != 0) else 0 for i in range(max_len)] for seq in padded_sequences])

def compute_loss(logits, labels, mask):
    num_classes = logits.shape[-1]
    smooth_positives = 1.0 - LABEL_SMOOTHING
    smooth_negatives = LABEL_SMOOTHING / num_classes
    onehot_labels = jax.nn.one_hot(labels, num_classes)
    smooth_labels = onehot_labels * smooth_positives + smooth_negatives

    loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits + EPSILON, axis=-1), axis=-1)
    loss = (loss * mask).sum() / (mask.sum() + EPSILON)
    return loss

@partial(jax.pmap, axis_name='batch')
def train_step(state, batch, mask, init_state, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits, new_state = state.apply_fn(
            params, batch[:, :-1], init_state, 
            deterministic=False, 
            rngs={'dropout': dropout_rng}
        )
        loss = compute_loss(logits, batch[:, 1:], mask[:, 1:])
        return loss, (logits, new_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_state)), grads = grad_fn(state.params)
    
    # Check for NaN values
    is_nan = jnp.isnan(loss).any()
    
    # Replace NaN gradients with zeros
    grads = jax.tree_map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), grads)
    
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    new_state = state.apply_gradients(grads=grads)
    max_grad = jax.lax.pmean(jnp.max(jnp.abs(jax.tree_util.tree_leaves(grads)[0])), axis_name='batch')
    return new_state, loss, new_state, max_grad, is_nan, new_dropout_rng

def train():
    global global_step
    
    print("Initializing training state...")
    
    latest_checkpoint = find_latest_checkpoint(SAVE_PATH)
    if latest_checkpoint:
        checkpoint_file = os.path.join(latest_checkpoint, "checkpoint")
        print(f"Loading checkpoint from {checkpoint_file}")
        if os.path.exists(checkpoint_file):
            loaded_state = load_checkpoint(checkpoint_file)
            learning_rate_schedule = create_learning_rate_schedule()
            train_state = create_train_state(loaded_state['params'], learning_rate_schedule)
            global_step = loaded_state['step']
        else:
            print(f"Checkpoint file not found: {checkpoint_file}")
            print("Starting from scratch.")
            learning_rate_schedule = create_learning_rate_schedule()
            train_state = create_train_state(params, learning_rate_schedule)
            global_step = 0
    else:
        print("No checkpoint found. Starting from scratch.")
        learning_rate_schedule = create_learning_rate_schedule()
        train_state = create_train_state(params, learning_rate_schedule)
        global_step = 0
    
    train_state = jax.device_put_replicated(train_state, devices)
    
    print("Training state initialized.")

    dummy_batch = jnp.ones((num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN), dtype=jnp.int32)
    dummy_mask = jnp.ones((num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN), dtype=jnp.int32)
    dummy_init_state = RWKV.get_init_state(config, BATCH_SIZE_PER_DEVICE)
    dummy_init_state = jnp.repeat(dummy_init_state[jnp.newaxis, ...], num_devices, axis=0)
    dropout_rng = jax.random.PRNGKey(0)
    dropout_rng = jax.random.split(dropout_rng, num_devices)
    
    print("Compiling training step...")
    train_state, _, _, _, _, dropout_rng = train_step(train_state, dummy_batch, dummy_mask, dummy_init_state, dropout_rng)
    print("Compilation done.")

    total_steps = (len(dataset) // (BATCH_SIZE * SEQ_LEN)) * EPOCHS

    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(EPOCHS):
            steps_per_epoch = len(dataset) // (BATCH_SIZE * SEQ_LEN)
            for step in range(steps_per_epoch):
                rng = jax.random.PRNGKey(epoch * steps_per_epoch + step)

                idxs = jax.random.randint(rng, (BATCH_SIZE,), 0, len(dataset) - SEQ_LEN)
                sequences = [dataset[idx:idx+SEQ_LEN] for idx in idxs]

                padded_sequences = pad_sequences(sequences, SEQ_LEN)
                mask = create_mask(padded_sequences, SEQ_LEN)

                padded_sequences = padded_sequences.reshape(num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN)
                mask = mask.reshape(num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN)

                padded_sequences = jnp.array(padded_sequences)
                mask = jnp.array(mask)

                init_state = RWKV.get_init_state(config, BATCH_SIZE_PER_DEVICE)
                init_state = jnp.repeat(init_state[jnp.newaxis, ...], num_devices, axis=0)

                train_state, loss, _, max_grad, is_nan, dropout_rng = train_step(train_state, padded_sequences, mask, init_state, dropout_rng)
                
                global_step += 1
                pbar.update(1)
                
                if global_step % 2 == 0:
                    loss = jax.device_get(loss)
                    mean_loss = jnp.mean(loss)
                    max_grad = jax.device_get(max_grad)
                    is_nan = jax.device_get(is_nan)
                    
                    if is_nan.any():
                        print(f"Warning: NaN detected at step {global_step}")
                    elif jnp.isinf(mean_loss):
                        print(f"Warning: Inf loss detected at step {global_step}")
                    else:
                        print(f"Step {global_step}, Loss: {mean_loss:.4f}, Max gradient: {max_grad:.4f}")

                if global_step % SAVE_EVERY == 0:
                    save_dir = os.path.join(SAVE_PATH, f"checkpoint_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    flat_params = jax.tree_util.tree_map(lambda x: x[0], jax.device_get(train_state.params))
                    checkpoint_file = os.path.join(save_dir, "checkpoint")
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'params': flat_params, 'step': global_step}, f)

    final_save_dir = os.path.join(SAVE_PATH, "checkpoint_final")
    os.makedirs(final_save_dir, exist_ok=True)
    flat_params = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x, jax.device_get(train_state.params))
    final_checkpoint_file = os.path.join(final_save_dir, "checkpoint")
    with open(final_checkpoint_file, 'wb') as f:
        pickle.dump({'params': flat_params, 'step': global_step}, f)

if __name__ == "__main__":
    train()
