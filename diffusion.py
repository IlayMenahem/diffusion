import optax
import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm import tqdm

from model import mnist_unet
from dataloader import batch_size, dataloader_train, dataloader_test
from utils import Accumulator

@eqx.filter_jit
def loss_fn(model, x, label, t, img):
    output = jax.vmap(model)(x, label, t)
    mse = jnp.mean(optax.losses.squared_error(output, img))
    loss = mse

    return loss

@eqx.filter_jit
def batch_train(model, data, optimizer, optimizer_state, loss_fn):
    x, label, t, img = data
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, label, t, img)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    return model, optimizer_state, loss

def epoch_train(model, train_loader, optimizer, optimizer_state, loss_fn):
    progress_bar = tqdm(total=len(train_loader._data_source)//batch_size)

    for step, data in enumerate(train_loader):
        model, optimizer_state, loss = batch_train(model, data, optimizer, optimizer_state, loss_fn)

        progress_bar.update(1)
        progress_bar.set_description(f'Loss: {loss:.4f}')

    return model, optimizer_state

def validate(model, loader, loss_fn):
    model = eqx.nn.inference_mode(model)
    progress_bar = tqdm(total=len(loader._data_source)//batch_size)
    loss_accumulator = Accumulator()

    for step, (x, label, t, img) in enumerate(loader):
        loss_accumulator.update(loss_fn(model, x, label, t, img))

        progress_bar.update(1)
        progress_bar.set_description(f'Loss: {loss_accumulator()}')

    avg_loss = loss_accumulator()

    return avg_loss

def train_model(model, train_loader, val_loader, num_epochs, optimizer, optimizer_state, loss_fn):
    val_loss_prev = float('inf')

    for epoch in range(num_epochs):
        model, optimizer_state = epoch_train(model, train_loader, optimizer, optimizer_state, loss_fn)

        val_loss = validate(model, val_loader, loss_fn)

        if val_loss < val_loss_prev:
            eqx.tree_serialise_leaves('diffusion_model.eqx', model)

        val_loss_prev = val_loss

if __name__ == '__main__':
    num_epochs = 10
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    model = mnist_unet(subkey)

    learning_rate = 1e-4
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_fn = loss_fn

    train_model(model, dataloader_train, dataloader_test, num_epochs, optimizer, optimizer_state, loss_fn)
