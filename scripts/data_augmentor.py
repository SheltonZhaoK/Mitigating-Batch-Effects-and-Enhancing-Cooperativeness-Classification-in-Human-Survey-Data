import logging

import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

def create_synthetic_gan(subset_data, label):
        model = Gan(data=subset_data)
        generator = model._generator()
        discriminator = model._discriminator()
        gan_model = model._GAN(generator=generator, discriminator=discriminator)
        trained_model = model.train(generator=generator, discriminator=discriminator, gan=gan_model)
        new_data_subset = pd.DataFrame(trained_model.predict(subset_data), columns=subset_data.columns.to_list())
        return pd.concat([subset_data, new_data_subset]), pd.concat([label, label])

def augment_data(data, label, target, args, configs):
    if args.a == "smote":
        data, target = SMOTE(random_state=configs["seed"]).fit_resample(data, target)
    elif args.a == "editNN":
        data, target = EditedNearestNeighbours().fit_resample(data, target)
    elif args.a == "tomkLink":
        data, target = TomekLinks().fit_resample(data, target)
    elif args.a == "smoteNN":
        data, target = SMOTEENN(random_state=configs["seed"]).fit_resample(data, target)
    elif args.a == "smoteTomek":
        data, target = SMOTETomek(random_state=configs["seed"]).fit_resample(data, target)
    
    if args.gan:
        augmented_data_0, augmented_label_0 = create_synthetic_gan(data[target[label] == 0], target[target[label] == 0])
        augmented_data_1, augmented_label_1 = create_synthetic_gan(data[target[label] == 1], target[target[label] == 1])

        combined_synthesized_data = pd.concat([augmented_data_0, augmented_data_1])
        combined_label = pd.concat([augmented_label_0, augmented_label_1])

        data = pd.concat([data, combined_synthesized_data])
        target = pd.concat([target, combined_label])  
        # data = combined_synthesized_data
        # target = combined_label

        data = data.reset_index(drop=True)
        target = target.reset_index(drop=True)
    return data, target

# https://samanemami.github.io/
class Gan():
    def __init__(self, data):
        self.data = data
        self.n_epochs = 200

    # Genereta random noise in a latent space
    def _noise(self):
        noise = np.random.normal(0, 1, self.data.shape)
        return noise

    def _generator(self):
        model = tf.keras.Sequential(name="Generator_model")
        model.add(tf.keras.layers.Dense(15, activation='relu',
                                        kernel_initializer='he_uniform',
                                        input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(30, activation='relu'))
        model.add(tf.keras.layers.Dense(
            self.data.shape[1], activation='linear'))
        return model

    def _discriminator(self):
        model = tf.keras.Sequential(name="Discriminator_model")
        model.add(tf.keras.layers.Dense(25, activation='relu',
                                        kernel_initializer='he_uniform',
                                        input_dim=self.data.shape[1]))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        # sigmoid => real or fake
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # define the combined generator and discriminator model,
    # for updating the generator
    def _GAN(self, generator, discriminator):
        discriminator.trainable = False
        generator.trainable = True
        model = tf.keras.Sequential(name="GAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    # train the generator and discriminator
    def train(self, generator, discriminator, gan):

        # determine half the size of one batch, for updating the  discriminator
        # manually enumerate epochs
        for epoch in range(self.n_epochs):
            
            # Train the discriminator
            generated_data = generator.predict(self._noise())
            labels = np.concatenate([np.ones(self.data.shape[0]), np.zeros(self.data.shape[0])])
            X = np.concatenate([self.data, generated_data])
            discriminator.trainable = True
            d_loss , _ = discriminator.train_on_batch(X, labels)
            # Train the generator
            noise = self._noise()
            g_loss = gan.train_on_batch(noise, np.ones(self.data.shape[0]))
            print('>%d, d1=%.3f, d2=%.3f' %(epoch+1, d_loss, g_loss))

        return generator