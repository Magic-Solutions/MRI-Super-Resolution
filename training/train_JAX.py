import os
import nibabel as nib
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm

# Import the UNet3D model from models.Unet_JAX
from models.Unet_JAX import UNet3D

class MRIDataset:
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.nii')])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.nii')])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_img = nib.load(self.lr_files[idx]).get_fdata()
        hr_img = nib.load(self.hr_files[idx]).get_fdata()

        # Convert to JAX arrays and add a channel dimension
        lr_img = jnp.expand_dims(jnp.array(lr_img, dtype=jnp.float32), axis=0)
        hr_img = jnp.expand_dims(jnp.array(hr_img, dtype=jnp.float32), axis=0)

        return lr_img, hr_img

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 1, 256, 256, 150]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, lr_img, hr_img):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, lr_img)
        loss = jnp.mean((preds - hr_img) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, lr_img, hr_img):
    preds = state.apply_fn({'params': state.params}, lr_img)
    loss = jnp.mean((preds - hr_img) ** 2)
    return loss

def train_epoch(state, dataloader):
    epoch_loss = 0
    for lr_img, hr_img in tqdm(dataloader):
        state, loss = train_step(state, lr_img, hr_img)
        epoch_loss += loss * lr_img.shape[0]
    epoch_loss /= len(dataloader)
    return state, epoch_loss

def validate_epoch(state, dataloader):
    epoch_loss = 0
    for lr_img, hr_img in dataloader:
        loss = eval_step(state, lr_img, hr_img)
        epoch_loss += loss * lr_img.shape[0]
    epoch_loss /= len(dataloader)
    return epoch_loss

def main():
    # Paths to the low-res and high-res data
    lr_dir = 'data/train/low_res'
    hr_dir = 'data/train/high_res'

    # Dataset and DataLoader
    train_dataset = MRIDataset(lr_dir, hr_dir)
    train_loader = [(train_dataset[i][0], train_dataset[i][1]) for i in range(len(train_dataset))]

    # Model, criterion, optimizer
    rng = jax.random.PRNGKey(0)
    model = UNet3D()
    learning_rate = 1e-4
    state = create_train_state(rng, model, learning_rate)

    # Train the model
    num_epochs = 100
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        state, train_loss = train_epoch(state, train_loader)
        val_loss = validate_epoch(state, train_loader)  # Simplified, using train_loader for validation
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoints.save_checkpoint('./', state, epoch)
            print(f'Saved model with validation loss: {val_loss:.4f}')

if __name__ == '__main__':
    main()