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

class mnist_unet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    label_embedding: eqx.nn.Embedding
    time_embedding: eqx.nn.Embedding

    conv4: eqx.nn.ConvTranspose2d
    conv5: eqx.nn.ConvTranspose2d
    conv6: eqx.nn.ConvTranspose2d

    def __init__(self, key):
        key1, key2, key3, key4, key5, key6, key7, key8, key9, key10 = jax.random.split(key, 10)

        self.conv1 = eqx.nn.Conv2d(1, 16, 3, 1, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(16, 32, 3, 1, padding=1, key=key2)
        self.conv3 = eqx.nn.Conv2d(32, 64, 3, 1, padding=1, key=key3)

        self.linear1 = eqx.nn.Linear(64*(image_dim-1)**2, 128, key=key4)
        self.linear2 = eqx.nn.Linear(128, 64*image_dim**2, key=key5)

        self.label_embedding = eqx.nn.Embedding(10, 128, key=key6)
        self.time_embedding = eqx.nn.Embedding(50, 128, key=key7)

        self.conv4 = eqx.nn.ConvTranspose2d(128, 64, 3, 1, padding=1, key=key8)
        self.conv5 = eqx.nn.ConvTranspose2d(64, 32, 3, 1, padding=1, key=key9)
        self.conv6 = eqx.nn.ConvTranspose2d(32, 1, 3, 1, padding=1, key=key10)

    @jax.jit
    def __call__(self, x, label, time):
        label_encoding = self.label_embedding(label)
        time_encoding = self.time_embedding(time)

        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x3 = jax.nn.relu(self.conv3(x))

        x = eqx.nn.MaxPool2d(kernel_size=2)(x3)
        x = jnp.ravel(x)

        x = jax.nn.relu(self.linear1(x)+label_encoding+time_encoding)
        x = jax.nn.relu(self.linear2(x))

        x = jnp.reshape(x, (64, image_dim, image_dim))
        x4_input = jnp.concatenate([x, x3])

        x = jax.nn.relu(self.conv4(x4_input))
        x = jax.nn.relu(self.conv5(x))
        x = jax.nn.sigmoid(self.conv6(x))

        return x
