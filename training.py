from art.attacks import EvasionAttack
from tensorflow.keras import Model
from typing import *
import tensorflow as tf

from encoders import *
import numpy as np
from abc import abstractmethod


class TrainingPipeline:

    @abstractmethod
    def generate_attack(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train_autoencoder(self, attack: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self,filepath: str) ->None:
        pass


class SimilarityEncoderTraining(TrainingPipeline, Encoder):

    def __init__(self, attack_generator: EvasionAttack, *args, **kwargs):
        super(SimilarityEncoderTraining, self).__init__(*args, **kwargs)
        self.attack_generator = attack_generator

    def generate_attack(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.attack_generator.generate(x, y)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.generate_attack(inputs)
        return super().call(x)

    def save_model(self, filepath: str) -> None:
        tf.saved_model.save(super(), filepath)


class SimilarityEncoderTrainingv2(TrainingPipeline):

    def __init__(self, attack_generator: EvasionAttack, autoencoder: Encoder, *args, **kwargs):
        self.attack_generator = attack_generator
        self.autoencoder = autoencoder.compile()
        self.history = None

    def generate_attack(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.attack_generator.generate(x, y)

    def train_autoencoder(self, x: np.ndarray, y: np.ndarray) -> None:
        x = self.generate_attack(x, y)
        history = self.autoencoder.fit(x, y)


    def history(self):
        return self.history

    def save_model(self,filepath: str) ->None:
        tf.saved_model.save(self.autoencoder, filepath)




