import os
import pickle
import numpy as np
import tensorflow as tf
from FeaturesExtractor import FeaturesExtractor

class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path = males_files_path
        self.features_extractor = FeaturesExtractor()

    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        # collect voice features
        female_voice_features = self.collect_features(females)
        male_voice_features = self.collect_features(males)
        
        # Combine features and labels
        X = np.vstack((female_voice_features, male_voice_features))
        y = np.hstack((np.zeros(len(female_voice_features)), np.ones(len(male_voice_features))))
        
        # Shuffle data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Create and train neural network
        model = self.build_neural_network()
        model.fit(X, y, epochs=10, batch_size=32)
        
        # Save model
        model.save("gender_classification_model")

    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path)]
        males = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path)]
        return females, males

    def collect_features(self, files):
        """
        Collect voice features from various speakers of the same gender.

        Args:
            files (list) : List of voice file paths.

        Returns:
            (array) : Extracted features matrix.
        """
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%-10s %-10s" % ("PROCESSING", file))
            # extract MFCC & delta MFCC features from audio
            vector = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        return features

    def build_neural_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(39,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model
if __name__ == "__main__":
    models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    models_trainer.process()
