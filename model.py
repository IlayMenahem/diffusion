import equinox as eqx
import jax
import jax.numpy as jnp

image_dim = 28

class mnist_model(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(1, 32, 3, 1, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(32, 64, 3, 1, padding=1, key=key2)
        self.linear1 = eqx.nn.Linear(64*(image_dim-1)**2, 128, key=key3)
        self.linear2 = eqx.nn.Linear(128, 10, key=key4)

    @jax.jit
    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = eqx.nn.MaxPool2d(kernel_size=2)(x)
        x = jnp.ravel(x)
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        x = jax.nn.log_softmax(x)

        return x

class mnist_feature_extractor(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(1, 32, 3, 1, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(32, 64, 3, 1, padding=1, key=key2)
        self.linear1 = eqx.nn.Linear(64*(image_dim-1)**2, 128, key=key3)
        self.linear2 = eqx.nn.Linear(128, 128, key=key4)

    @jax.jit
    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = eqx.nn.MaxPool2d(kernel_size=2)(x)
        x = jnp.ravel(x)
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)

        return x

class mnist_generator(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    conv1: eqx.nn.ConvTranspose2d
    conv2: eqx.nn.ConvTranspose2d

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.linear1 = eqx.nn.Linear(128, 128, key=key1)
        self.linear2 = eqx.nn.Linear(128, 64*image_dim**2, key=key2)
        self.conv1 = eqx.nn.ConvTranspose2d(64, 32, 3, 1, padding=1, key=key3)
        self.conv2 = eqx.nn.ConvTranspose2d(32, 1, 3, 1, padding=1, key=key4)

    @jax.jit
    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        x = x.reshape((64, image_dim, image_dim))
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.sigmoid(self.conv2(x))

        return x

class mnist_diffusion(eqx.Module):
    feature_extractor: mnist_feature_extractor
    label_embedding: eqx.nn.Embedding
    generator: mnist_generator
    time_embedding: eqx.nn.Embedding

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.feature_extractor = mnist_feature_extractor(key1)
        self.label_embedding = eqx.nn.Embedding(10, 128, key=key2)
        self.generator = mnist_generator(key3)
        self.time_embedding = eqx.nn.Embedding(50, 128, key=key4)

    @jax.jit
    def __call__(self, x, label, time):
        x = self.feature_extractor(x)
        label_embedding = self.label_embedding(label)
        time_embedding = self.time_embedding(time)
        x = x + label_embedding + time_embedding
        x = self.generator(x)

        return x
