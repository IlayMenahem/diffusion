import equinox as eqx
import jax
from model import mnist_unet
from utils import diffuse

if __name__ == '__main__':
    key = jax.random.key(0)

    model = mnist_unet(key)
    model = eqx.tree_deserialise_leaves('diffusion_model.eqx', model)

    for i in range(10):
        diffuse(model, i, 50)
