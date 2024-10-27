import os
import numpy as np
from sklearn.svm import SVC
from FeaturesExtractor import FeaturesExtractor

class GenderIdentifier:

    def _init_(self, females_model_path, males_model_path):
        self.females_model_path = females_model_path
        self.males_model_path = males_model_path
        self.features_extractor = FeaturesExtractor()
        # load models
        self.females_svm = self.load_model(females_model_path)
        self.males_svm = self.load_model(males_model_path)

    def process(self, females_testing_path, males_testing_path):
        females_files = self.get_file_paths(females_testing_path)
        males_files = self.get_file_paths(males_testing_path)
        total_samples = len(females_files) + len(males_files)
        correct_predictions = 0

        # Test females
        for file in females_files:
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))
            vector = self.features_extractor.extract_features(file)
            prediction = self.predict_gender(vector)
            print("%10s %6s %1s" % ("+ PREDICTION", ":", prediction))
            if prediction == "female":
                correct_predictions += 1

        # Test males
        for file in males_files:
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))
            vector = self.features_extractor.extract_features(file)
            prediction = self.predict_gender(vector)
            print("%10s %6s %1s" % ("+ PREDICTION", ":", prediction))
            if prediction == "male":
                correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100
        print("Accuracy:", accuracy, "%")

    def get_file_paths(self, directory):
        # Get file paths in the directory
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    def predict_gender(self, vector):
        # Predict gender using SVM classifiers
        is_female = self.females_svm.predict([vector])
        return "female" if is_female else "male"

    def load_model(self, model_path):
        # Load SVM model from file
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model


if _name_ == "_main_":
    identifier = GenderIdentifier("females.svm", "males.svm")
    identifier.process("TestingData/females", "TestingData/males")