import equinox as eqx
import jax
from model import mnist_diffusion
from utils import diffuse

if __name__ == '__main__':
    key = jax.random.key(0)

    model = mnist_diffusion(key)
    model = eqx.tree_deserialise_leaves('diffusion_model.eqx', model)

    for i in range(10):
        diffuse(model, i, 50)
