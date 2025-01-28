import optax
import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm import tqdm

from model import mnist_model
from dataloader import dataloader_train, dataloader_test, batch_size

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
def accuracy_fn(model, data, target):
    probs = jax.vmap(model)(data)
    preds = jnp.argmax(probs, axis=-1)
    accuracy = jnp.mean(preds == target)

    return accuracy

@eqx.filter_jit
def loss_fn(model, data, target):
    output = jax.vmap(model)(data)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(output, target))

    return loss

@eqx.filter_jit
def batch_train(model, data, target, optimizer, optimizer_state, loss_fn, accuracy_fn):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, data, target)
    accuracy = accuracy_fn(model, data, target)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    return model, optimizer_state, loss, accuracy

def epoch_train(model, train_loader, optimizer, optimizer_state, loss_fn, accuracy_fn):
    progress_bar = tqdm(total=len(train_loader._data_source)//batch_size)

    for step, (data, target) in enumerate(train_loader):
        model, optimizer_state, loss, accuracy = batch_train(model, data, target, optimizer, optimizer_state, loss_fn, accuracy_fn)

        progress_bar.update(1)
        progress_bar.set_description(f'Loss: {loss:.4f} - Accuracy: {accuracy:.4f}')

    return model, optimizer_state

def validate(model, loader, loss_fn, accuracy_fn):
    model = eqx.nn.inference_mode(model)
    progress_bar = tqdm(total=len(loader._data_source)//batch_size)
    loss_accumulator = Accumulator()
    accuracy_accumulator = Accumulator()

    for step, (data, target) in enumerate(loader):
        loss_accumulator.update(loss_fn(model, data, target))
        accuracy_accumulator.update(accuracy_fn(model, data, target))

        progress_bar.update(1)
        progress_bar.set_description(f'Loss: {loss_accumulator()} - Accuracy: {accuracy_accumulator()}')

def train_model(model, train_loader, val_loader, num_epochs, optimizer, optimizer_state, loss_fn, accuracy_fn):
    for epoch in range(num_epochs):
        model, optimizer_state = epoch_train(model, train_loader, optimizer, optimizer_state, loss_fn, accuracy_fn)

        validate(model, val_loader, loss_fn, accuracy_fn)

if __name__ == '__main__':
    num_epochs = 10
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    model = mnist_model(subkey)

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_fn = loss_fn
    accurecy_fn = accuracy_fn

    train_model(model, dataloader_train, dataloader_test, num_epochs, optimizer, optimizer_state, loss_fn, accurecy_fn)
