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

# Configuration
MODEL_PATH = './weight/RWKV-x060.pkl'
TOKENIZER_PATH = "rwkv_vocab_v20230424.txt"
DATA_PATH = 'data/minipile'
SAVE_PATH = os.path.abspath("rwkv-pretrain-checkpoint")

config = RWKVConfig(
    vocab_size=50277,
    n_layer=12,
    n_embd=768,
    dim_att=768,
    dim_ffn=3072,
    head_size_a=64,
    n_head=12,
    head_size_divisor=8,
    dropout=0.1,
    layer_norm_epsilon=1e-5,
    chunk_size=64,
    subchunk_size=32
)

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEQ_LEN = 512
EPOCHS = 10
SAVE_EVERY = 1000
global_step = 0

devices = jax.devices()
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
        model = RWKV(config)
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

def create_train_state(params, learning_rate):
    tx = optax.adam(learning_rate)
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
        if len(seq) > max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value))
    return np.array(padded, dtype=np.int32)

def create_mask(padded_sequences, max_len):
    return np.array([[1 if i < len(seq) else 0 for i in range(max_len)] for seq in padded_sequences])

@partial(jax.pmap, axis_name='batch')
def train_step(state, batch, mask, init_state):
    def loss_fn(params):
        logits, new_state = model.apply(params, batch[:, :-1], init_state)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1)
        )
        loss = (loss * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        return loss, (logits, new_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_state)), grads = grad_fn(state.params)
    
    # Compute average gradient across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, new_state

def train():
    global global_step
    
    print("Initializing training state...")
    
    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint(SAVE_PATH)
    if latest_checkpoint:
        checkpoint_file = os.path.join(latest_checkpoint, "checkpoint")
        print(f"Loading checkpoint from {checkpoint_file}")
        if os.path.exists(checkpoint_file):
            loaded_state = load_checkpoint(checkpoint_file)
            train_state = create_train_state(loaded_state['params'], LEARNING_RATE)
            global_step = loaded_state['step']
        else:
            print(f"Checkpoint file not found: {checkpoint_file}")
            print("Starting from scratch.")
            train_state = create_train_state(params, LEARNING_RATE)
            global_step = 0
    else:
        print("No checkpoint found. Starting from scratch.")
        train_state = create_train_state(params, LEARNING_RATE)
        global_step = 0
    
    train_state = jax.device_put_replicated(train_state, devices)
    
    print("Training state initialized.")

    dummy_batch = jnp.ones((num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN), dtype=jnp.int32)
    dummy_mask = jnp.ones((num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN), dtype=jnp.int32)
    dummy_init_state = RWKV.get_init_state(config, BATCH_SIZE_PER_DEVICE)
    dummy_init_state = jnp.repeat(dummy_init_state[jnp.newaxis, ...], num_devices, axis=0)

    print("Compiling training step...")
    train_step(train_state, dummy_batch, dummy_mask, dummy_init_state)
    print("Compilation done.")

    total_steps = (len(dataset) // (BATCH_SIZE * SEQ_LEN)) * EPOCHS

    for epoch in range(EPOCHS):
        steps_per_epoch = len(dataset) // (BATCH_SIZE * SEQ_LEN)
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
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

                train_state, loss, _ = train_step(train_state, padded_sequences, mask, init_state)

                global_step += 1
                pbar.update(1)

                if global_step % 100 == 0:
                    loss = jax.device_get(loss)
                    print(f"Step {global_step}, Loss: {loss.mean():.4f}")

                if global_step % SAVE_EVERY == 0:
                    save_dir = os.path.join(SAVE_PATH, f"checkpoint_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    flat_params = jax.tree_util.tree_map(lambda x: x[0], jax.device_get(train_state.params))
                    checkpoint_file = os.path.join(save_dir, "checkpoint")
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'params': flat_params, 'step': global_step}, f)

    final_save_dir = os.path.join(SAVE_PATH, "checkpoint_final")
    os.makedirs(final_save_dir, exist_ok=True)
    flat_params = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x, jax.device_get(train_state.params))
    final_checkpoint_file = os.path.join(final_save_dir, "checkpoint")
    with open(final_checkpoint_file, 'wb') as f:
        pickle.dump({'params': flat_params, 'step': global_step}, f)

if __name__ == "__main__":
    train()