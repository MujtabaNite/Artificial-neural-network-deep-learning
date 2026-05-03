import numpy as np  # NumPy provides fast mathematical operations on large arrays; essential for reshaping, scaling, and manipulating image tensors used in neural networks  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas (not used here but commonly for data handling)
import matplotlib.pyplot as plt  # Import Matplotlib for plotting generated images
import tensorflow as tf  # Import TensorFlow for building and training neural networks
from tensorflow.keras import layers, models  # Import Keras layers and model utilities
from pathlib import Path  # Import Path for handling directories and saving images

# 1) Load and preprocess MNIST correctly — MNIST is a dataset of 28x28 grayscale handwritten digits; perfect for GAN experiments because it’s simple yet meaningful
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Load MNIST dataset (images + labels)

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0  # Convert pixel values (0–255 integers) to 32-bit floats and scale to [0,1]; this stabilizes neural network training('float32') / 255.0  # Convert training images to float and normalize
x_test = x_test.astype('float32') / 255.0  # Convert test images to float and normalize

# Add channel dimension -> (N, 28, 28, 1)
# MNIST images are originally shaped (28,28). CNNs expect 4D tensors: (batch, height, width, channels). -> (N, 28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)  # Add a channel dimension for training images
x_test = np.expand_dims(x_test, axis=-1)  # Add a channel dimension for test images

# Hyperparameters
noise_dim = 100  # The generator takes a 100‑dimensional random noise vector — high‑dimensional noise captures richer variation for generated images  # Size of the random noise vector for generator
num_examples_to_generate = 16  # How many images to generate for preview
batch_size = 128  # Number of samples per training batch
epochs = 10  # Number of training epochs

# 2) Build networks

def build_generator():  # Generator transforms random noise into a synthetic 28x28 grayscale MNIST-like image():  # Generator network creates fake images from noise
    model = models.Sequential([
        layers.Dense(7 * 7 * 256, input_shape=(noise_dim,)),  # Expand noise vector into a large feature space that can be reshaped into a small image-like block, input_shape=(noise_dim,)),  # Fully connected layer to expand noise vector
        layers.BatchNormalization(),  # Normalize outputs to stabilize training
        layers.LeakyReLU(),  # Activation function allowing small negative gradients
        layers.Reshape((7, 7, 256)),  # Reshape vector into a 7x7x256 feature map

        # First deconvolution layer
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'),  # Start upsampling the 7x7 block using learned filters; Conv2DTranspose increases spatial dimensions(128, (5, 5), strides=(1, 1), padding='same'),  # Upsample feature map
        layers.BatchNormalization(),  # Normalize layer output
        layers.LeakyReLU(),  # Activation

        # Second deconvolution layer
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),  # Increase spatial size
        layers.BatchNormalization(),  # Normalize layer output
        layers.LeakyReLU(),  # Activation

        # Final layer to produce 28x28x1 output
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')  # Output grayscale image
    ])
    return model  # Return the generator model

def build_discriminator():  # Discriminator receives a 28x28 image and outputs a single logit — how real or fake the image looks():  # Discriminator distinguishes fake images from real
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),  # Convolution extracts low‑level patterns like strokes and edges from images, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),  # Extract features
        layers.LeakyReLU(),  # Activation for non-linearity
        layers.Dropout(0.3),  # Dropout to prevent overfitting

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),  # Deeper convolution layer
        layers.LeakyReLU(),  # Activation
        layers.Dropout(0.3),  # More dropout

        layers.Flatten(),  # Flatten feature maps
        layers.Dense(1)  # Final output: real (positive) or fake (negative)
    ])
    return model  # Return discriminator

generator = build_generator()  # Initialize generator
discriminator = build_discriminator()  # Initialize discriminator

generator_optimizer = tf.keras.optimizers.Adam(1e-4)  # Adam optimizer with a low learning rate to prevent unstable GAN training = tf.keras.optimizers.Adam(1e-4)  # Optimizer for generator
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)  # Optimizer for discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Loss function

def generator_loss(fake_output):  # Generator tries to fool the discriminator, so it wants the discriminator to output 'real' (1) for fake images(fake_output):  # Generator loss
    return cross_entropy(tf.ones_like(fake_output), fake_output)  # Wants discriminator to classify fake as real

def discriminator_loss(real_output, fake_output):  # Discriminator simultaneously learns to classify real as 1 and fake as 0 — two separate binary cross-entropy losses(real_output, fake_output):  # Discriminator loss
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # Loss on real images
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Loss on fake images
    return real_loss + fake_loss  # Combined loss

# 3) Image generation helper

def generate_and_save_images(model, epoch):  # Creates a preview grid of generated images for monitoring training progress epoch-by-epoch(model, epoch):  # Function to save sample images after each epoch
    noise = tf.random.normal([num_examples_to_generate, noise_dim])  # Generate noise vectors
    generated_images = model(noise, training=False)  # Produce fake images

    plt.figure(figsize=(4, 4))  # Create a 4x4 image grid
    for i in range(generated_images.shape[0]):  # Loop through generated images
        plt.subplot(4, 4, i + 1)  # Create subplot
        img = generated_images[i, :, :, 0]  # Select image
        plt.imshow(img * 255.0, cmap='gray')  # Convert normalized image back to grayscale
        plt.axis('off')  # Hide axes

    save_dir = Path("generated_images")  # Create directory for saving images
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    plt.tight_layout()  # Adjust spacing
    plt.savefig(save_dir / f"image_at_epoch_{epoch:04d}.png")  # Save figure
    plt.close()  # Close plot to free memory

# 4) Prepare dataset
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)  # Convert training images into a TensorFlow dataset object for efficient batching and shuffling.Dataset.from_tensor_slices(x_train)  # Convert training data into TF dataset
train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size)  # Shuffle + batch data

# 5) Training loop

def train(dataset, epochs):  # Main GAN training loop: trains discriminator first, then generator, for each batch and each epoch, epochs):  # Training function
    for epoch in range(epochs):  # Loop through epochs
        for image_batch in dataset:  # Loop through batches
            curr_batch_size = tf.shape(image_batch)[0]  # Get batch size

            # Train discriminator
            noise = tf.random.normal([curr_batch_size, noise_dim])  # Generate noise batch
            with tf.GradientTape() as disc_tape:  # GradientTape records operations for automatic differentiation; used here to compute discriminator gradients() as disc_tape:  # Track gradients
                generated_images = generator(noise, training=True)  # Create fake images

                real_output = discriminator(image_batch, training=True)  # Discriminator output on real
                fake_output = discriminator(generated_images, training=True)  # Discriminator evaluates how realistic the generator’s newly created images look(generated_images, training=True)  # Discriminator output on fake

                disc_loss = discriminator_loss(real_output, fake_output)  # Calculate discriminator loss

            grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)  # Compute gradient of discriminator loss w.r.t. all trainable weights.gradient(disc_loss, discriminator.trainable_variables)  # Compute gradients
            discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))  # Update discriminator

            # Train generator
            noise = tf.random.normal([curr_batch_size, noise_dim])  # New noise batch
            with tf.GradientTape() as gen_tape:  # Track generator gradients
                generated_images = generator(noise, training=True)  # Generate fake images
                fake_output = discriminator(generated_images, training=True)  # Discriminator evaluates fake
                gen_loss = generator_loss(fake_output)  # Compute generator loss

            grads = gen_tape.gradient(gen_loss, generator.trainable_variables)  # Compute gradients
            generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))  # Update generator

        print(f'Epoch {epoch + 1}, Generator Loss: {gen_loss.numpy():.4f}, Discriminator Loss: {disc_loss.numpy():.4f}')  # Print losses so you can track which network is outperforming the other(f'Epoch {epoch + 1}, Generator Loss: {gen_loss.numpy():.4f}, Discriminator Loss: {disc_loss.numpy():.4f}')  # Log losses
        generate_and_save_images(generator, epoch + 1)  # Save sample images per epoch

# Run training
train(train_dataset, epochs)  # Start training
