import equinox as eqx
import jax
from model import mnist_unet
import matplotlib.pyplot as plt

def diffuse(model, label, max_t):
    guassian_noise = jax.random.normal(jax.random.PRNGKey(0), (1, 28, 28))
    img = model(guassian_noise, label, max_t)

    plt.figure(figsize=(5, 5))
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f'Diffusion of label {label}')
    plt.show()

if __name__ == '__main__':
    key = jax.random.key(0)

    model = mnist_unet(key)
    model = eqx.tree_deserialise_leaves('diffusion_model.eqx', model)

    for i in range(10):
        diffuse(model, i, 50)
