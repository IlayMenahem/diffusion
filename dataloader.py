import grain.python as pygrain
import jax.numpy as jnp
from torchvision.datasets import MNIST
import jax
import optax

def get_alpha_bar(t, max_t):
    schedule = optax.schedules.cosine_decay_schedule(1.0, max_t)
    betas = jax.vmap(schedule)(jnp.arange(max_t))
    alphas = 1 - betas
    alpha_bar = jnp.cumprod(alphas)[-1]

    return alpha_bar

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
        alpha_bar = get_alpha_bar(t, self.max_t)

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
