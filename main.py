from submissionParser import SubmissionParser
from vectorizer import Vectorizer
import numpy as np


def generate_distance_matrix():
    def cosine_similarity(m1, m2):
        norm1 = np.linalg.norm(m1, axis=1)
        norm2 = np.linalg.norm(m2, axis=1)

        dot = np.sum(m1 * m2, axis=1)
        return dot / (norm1 * norm2)

    def similarity_test(produced_similarity, vector_dataset):
        distances_check = []
        for i in range(len(vector_dataset)):
            distances_check.append(
                np.dot(vector_dataset[i, :, 0], vector_dataset[i, :, 1]) /
                (np.linalg.norm(vector_dataset[i, :, 0]) * np.linalg.norm(vector_dataset[i, :, 1])))

        print('distance correct:', np.allclose(
            produced_similarity, distances_check))

    parser = SubmissionParser(
        "ClinVarVCVRelease_2024-02.xml", "csv/", 100000, '~')
    dataset = parser.parse()
    dataset = np.unique(dataset, axis=0)
    parser.save_dataset(dataset)
    print('parsed')

    vectorizer = Vectorizer('BAAI/bge-large-en-v1.5')
    vectors = vectorizer.vectorize(dataset)
    # print('vectorized shape:', vectorizer.vectorized.shape)
    vectorizer.save_dataset('./csv/', vectors, dataset)
    # vectorizer.load_dataset(
    #   vectorized_filename='./csv/submission_embeddings.npy')
    print('vectorized')

    similarity = cosine_similarity(
        vectors[:, :, 0], vectors[:, :, 1])

    similarity_sorted = np.argsort(similarity)
    sorted_submissions = dataset[similarity_sorted]
    sorted_submissions = np.hstack((
        sorted_submissions, similarity[similarity_sorted][:, None]))

    np.save('./results/sorted_submissions.npy', sorted_submissions)

    # top_100 = np.argpartition(distances, -100)[-100:]
    # bottom_100 = np.argpartition(distances, 100)[:100]
    #
    # top_100 = vectorizer.dataset[top_100]
    # bottom_100 = vectorizer.dataset[bottom_100]
    #
    # np.save('./results/top_100.npy', top_100)
    # np.save('./results/bottom_100.npy', bottom_100)


def analyze():

    sorted_submissions = np.load('./results/sorted_submissions.npy')
    print(sorted_submissions[0:10, :])
    print(sorted_submissions[-10:, :])

    print('classification averages')
    unique_classifications = np.unique(sorted_submissions[:, 0])
    print(unique_classifications)
    for classification in unique_classifications:
        class_indices = np.where(sorted_submissions[:, 0] == classification)
        c_similarities = sorted_submissions[class_indices][:, 3]
        avg = np.mean(c_similarities.astype(np.float64))
        print(classification, avg)


generate_distance_matrix()
analyze()
