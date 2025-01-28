import grain.python as pygrain
import jax.numpy as jnp
from torchvision.datasets import MNIST

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
batch_size = 128

mnist_dataset_train = Dataset(data_dir, train=True)
mnist_dataset_test = Dataset(data_dir, train=False)

sampler_train = pygrain.SequentialSampler(num_records=len(mnist_dataset_train),shard_options=pygrain.NoSharding())
sampler_test = pygrain.SequentialSampler(num_records=len(mnist_dataset_test),shard_options=pygrain.NoSharding())

dataloader_train = pygrain.DataLoader(data_source=mnist_dataset_train, sampler=sampler_train, operations=[pygrain.Batch(batch_size)])
dataloader_test = pygrain.DataLoader(data_source=mnist_dataset_test, sampler=sampler_test, operations=[pygrain.Batch(batch_size)])
