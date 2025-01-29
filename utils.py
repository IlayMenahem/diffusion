import matplotlib.pyplot as plt
import jax

class Accumulator:
    def __init__(self):
        self.data = 0
        self.count = 0

    def update(self, data):
        self.data += data
        self.count += 1

    def __call__(self):
        return self.data/self.count

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
        ax[1, i].set_title(f'Corrupted t is {t[i]}')

        ax[2, i].imshow(recu[i].reshape(28, 28), cmap='gray')
        ax[2, i].axis('off')
        ax[2, i].set_title('Reconstruction')

    plt.show()
