from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import json


def sparsity(matrix):
    return len(np.where(matrix == 0)[0]) / (matrix.shape[0] * matrix.shape[1])


def compare_vectorizers():
    texts = []
    with open('core/core_tokenized', encoding='utf-8') as file:
        texts = file.readlines()

    vectorizers = [
        lambda ng: CountVectorizer(binary=True, ngram_range=ng),
        lambda ng: CountVectorizer(ngram_range=ng),
        lambda ng: TfidfVectorizer(smooth_idf=False, sublinear_tf=False, ngram_range=ng),
        lambda ng: TfidfVectorizer(smooth_idf=False, sublinear_tf=True, ngram_range=ng),
        lambda ng: TfidfVectorizer(smooth_idf=True, sublinear_tf=False, ngram_range=ng),
        lambda ng: TfidfVectorizer(smooth_idf=True, sublinear_tf=True, ngram_range=ng),
    ]
    vectorizer_names = [
        'onehot',
        'Bag of Words',
        'TF-IDF',
        'TF-IDF sublinear',
        'TF-IDF smooth',
        'TF-IDF smooth sublinear',
    ]

    ngram_ranges = [
        (1, 1), (2, 2), (3, 3),
        (1, 2), (2, 3), (1, 3)
    ]

    csv_string = 'vectorizer;vocabulary size;matrix size;matrix sparsity\n'

    for i in range(len(vectorizers)):
        for ngram_range in ngram_ranges:
            print(f'{vectorizer_names[i]} {ngram_range}')
            vectorizer = vectorizers[i](ngram_range)
            x = vectorizer.fit_transform(texts)
            voc_size = len(vectorizer.get_feature_names_out())
            matr = x.toarray()
            csv_string += f'{vectorizer_names[i]} {ngram_range};{voc_size};{matr.shape[0]}x{matr.shape[1]};{round(sparsity(matr), 3)}\n'
    with open('output/classical_vectorizers.csv', 'w') as file:
        file.write(csv_string)


if __name__ == '__main__':
    # compare_vectorizers()
    texts = []
    with open('core/core_tokenized', encoding='utf-8') as file:
        texts = file.readlines()
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)
    print(vectorizer.get_feature_names_out())