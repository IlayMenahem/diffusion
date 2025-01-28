import optax
import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import mnist_diffusion
from dataloader import batch_size, dataloader_train, dataloader_test

class Accumulator:
    def __init__(self):
        self.data = 0
        self.count = 0

    def update(self, data):
        self.data += data
        self.count += 1

    def __call__(self):
        return self.data/self.count

@eqx.filter_jit
def loss_fn(model, x, label, t, img):
    output = jax.vmap(model)(x, label, t)
    mse = jnp.mean(optax.losses.squared_error(output, img))
    sharpness = jnp.mean(-jnp.abs(output-0.5))
    loss = mse + 0.1*sharpness

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

def display_batch(model, x, label, t, img, length=5):
    '''
    display the first `length` images in the batch, both the original, corrupted and the reconstruction
    '''
    recu = jax.vmap(model)(x, label, t)

    fig, ax = plt.subplots(3, length, figsize=(15, 5))

    for i in range(length):
        ax[0, i].imshow(img[i].reshape(28, 28), cmap='gray')
        ax[0, i].axis('off')
        ax[0, i].set_title('Original')

        ax[1, i].imshow(x[i].reshape(28, 28), cmap='gray')
        ax[1, i].axis('off')
        ax[1, i].set_title('Corrupted')

        ax[2, i].imshow(recu[i].reshape(28, 28), cmap='gray')
        ax[2, i].axis('off')
        ax[2, i].set_title('Reconstruction')

    plt.show()

def validate(model, loader, loss_fn):
    model = eqx.nn.inference_mode(model)
    progress_bar = tqdm(total=len(loader._data_source)//batch_size)
    loss_accumulator = Accumulator()

    for step, (x, label, t, img) in enumerate(loader):
        loss_accumulator.update(loss_fn(model, x, label, t, img))

        if step == 0:
            display_batch(model, x, label, t, img)

        progress_bar.update(1)
        progress_bar.set_description(f'Loss: {loss_accumulator()}')

def train_model(model, train_loader, val_loader, num_epochs, optimizer, optimizer_state, loss_fn):
    for epoch in range(num_epochs):
        validate(model, val_loader, loss_fn)
        model, optimizer_state = epoch_train(model, train_loader, optimizer, optimizer_state, loss_fn)


if __name__ == '__main__':
    num_epochs = 10
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    model = mnist_diffusion(subkey)

    learning_rate = 1e-4
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_fn = loss_fn

    train_model(model, dataloader_train, dataloader_test, num_epochs, optimizer, optimizer_state, loss_fn)
