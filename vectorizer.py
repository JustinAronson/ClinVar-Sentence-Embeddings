from sentence_transformers import SentenceTransformer
import numpy as np
import pickle


class Vectorizer:
    def __init__(self, model_name, dataset=None):
        self.model = SentenceTransformer(model_name)
        if dataset is not None:
            self.dataset = np.unique(dataset, axis=0)

    def encode_pathogenicity(self, dataset):
        pathogenicities = np.unique(dataset[:, 1])
        pathogenicity_encodings = self.model.encode(list(pathogenicities))

        pathogenicity_dict = {pathogenicity: encoding for pathogenicity, encoding in zip(
            pathogenicities, pathogenicity_encodings)}

        return np.array([pathogenicity_dict[pathogenicity]
                         for pathogenicity in dataset[:, 1]])

    def vectorize(self, dataset):
        pathogenicity_encodings = self.encode_pathogenicity(dataset)
        # pathogenicity_encodings = self.model.encode(list(self.dataset[:, 1]))
        encodings = self.model.encode(list(dataset[:, 2]))
        return np.stack(
            (pathogenicity_encodings, encodings), axis=2)

    def load_dataset(self, filename=None, vectorized_filename=None):
        if filename is not None:
            # self.dataset = np.genfromtxt(filename, delimiter=delim, dtype=str)
            self.dataset = np.load(filename)
            # Remove duplicate rows
            self.dataset = np.unique(self.dataset, axis=0)
            print(self.dataset.shape)

        if vectorized_filename is not None:
            self.vectorized = np.load(vectorized_filename)

    def save_dataset(self, filedir, vectorized, dataset):
        np.save(filedir + 'submission_embeddings.npy', vectorized)
        np.save(filedir + 'submission_labels.npy', dataset[0, :])


# vectorizer = Vectorizer('BAAI/bge-large-en-v1.5')
# vectorizer.load_dataset('./csv/submissions.npy')
# vectorizer.vectorize()
# print(vectorizer.vectorized.shape)
# vectorizer.save_dataset('./csv/')
