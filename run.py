import os
import matplotlib.pyplot as plt

from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

import tensorflow_datasets as tfds
import datasets.sr_dataset

#%matplotlib inline

# Location of model weights (needed for demo)
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

# Datasets

#div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
#div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

#train_ds = div2k_train.dataset(batch_size=4, random_transform=True)
#valid_ds = div2k_valid.dataset(batch_size=4, random_transform=True, repeat_count=1)

# Load dataset
train_ds = tfds.load('sr_dataset', split='train', as_supervised=True, shuffle_files=True)
valid_ds = tfds.load('sr_dataset', split='test', as_supervised=True, shuffle_files=True)

train_ds = train_ds.shuffle(buffer_size=100).batch(4)
valid_ds = valid_ds.shuffle(buffer_size=100).batch(4)

# Generator pre-training

model = generator()
model.summary()
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')
pre_trainer.train(train_ds,
                  valid_ds.take(10),
                  steps=80,
                  evaluate_every=10,
                  save_best_only=False)

pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

# Generator fine-tuning (GAN)

#gan_generator = generator()
#gan_generator.load_weights(weights_file('pre_generator.h5'))

#gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
#gan_trainer.train(train_ds, steps=200000)

#gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
#gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))

# Demo
#pre_generator = generator()
#gan_generator = generator()

#pre_generator.load_weights(weights_file('pre_generator.h5'))
#gan_generator.load_weights(weights_file('gan_generator.h5'))

from model import resolve_single
from utils import load_image

def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)

    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)

    plt.figure(figsize=(20, 20))

    images = [lr, pre_sr, gan_sr]
    titles = ['LR', 'SR (PRE)', 'SR (GAN)']
    positions = [1, 3, 4]

    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.savefig('test.png')

#resolve_and_plot('demo/0869x4-crop.png')