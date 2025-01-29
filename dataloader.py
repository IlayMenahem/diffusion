import grain.python as pygrain
import jax.numpy as jnp
from torchvision.datasets import MNIST
import jax

def schedule(t, max_t, s=0.008):
    max_t = jnp.linspace(0, 1, max_t)
    f_t = jnp.cos((max_t + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar_t = f_t / f_t[0]
    beta_t = 1 - alpha_bar_t[1:] / alpha_bar_t[:-1]
    beta_t = jnp.clip(beta_t, 0, 0.999)
    beta_t = jnp.concatenate([jnp.array([0.0]), beta_t])

    alpha_t = 1 - beta_t
    alpha_bar_t = jnp.cumprod(alpha_t)[t]

    return alpha_bar_t

class DiffusionDataset:
    def __init__(self, data_dir, key, max_t, train):
        self.data_dir = data_dir
        self.train = train
        self.key = key
        self.max_t = max_t
        self.load_data()

    def load_data(self):
        self.dataset = MNIST(self.data_dir, download=True, train=self.train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = jnp.array(img, dtype=jnp.float32)/255.0
        img = jnp.expand_dims(img, axis=0)
        label = jnp.array(label, dtype=jnp.int32)

        self.key, key1, key2 = jax.random.split(self.key, 3)
        t = jax.random.randint(key1, shape=(), minval=0, maxval=self.max_t)
        gaussian_noise = jax.random.normal(key2, img.shape)
        alpha_bar = schedule(t, self.max_t)

        x = jnp.sqrt(alpha_bar)*img + jnp.sqrt(1-alpha_bar)*gaussian_noise

        return x, label, t, img

class Dataset:
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.load_data()

    def load_data(self):
        self.dataset = MNIST(self.data_dir, download=True, train=self.train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = jnp.array(img, dtype=jnp.float32)/255.0
        img = jnp.expand_dims(img, axis=0)
        label = jnp.array(label, dtype=jnp.int32)

        return img, label

data_dir = '/tmp/mnist_dataset'
batch_size = 256
max_t = 50

diffusion_dataset_train = DiffusionDataset(data_dir, jax.random.key(0), max_t, True)
diffusion_dataset_test = DiffusionDataset(data_dir, jax.random.key(0), max_t, False)

sampler_train = pygrain.SequentialSampler(num_records=len(diffusion_dataset_train),shard_options=pygrain.NoSharding())
sampler_test = pygrain.SequentialSampler(num_records=len(diffusion_dataset_test),shard_options=pygrain.NoSharding())

dataloader_train = pygrain.DataLoader(data_source=diffusion_dataset_train, sampler=sampler_train, operations=[pygrain.Batch(batch_size)])
dataloader_test = pygrain.DataLoader(data_source=diffusion_dataset_test, sampler=sampler_test, operations=[pygrain.Batch(batch_size)])
