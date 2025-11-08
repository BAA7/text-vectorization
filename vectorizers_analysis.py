from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import json
from size_shrinking import find_n_components
from classical_vectorizers import sparsity
import tracemalloc
import time


def group_texts_by_tags(text_tags, tags):
    res = {}
    for tag in tags:
        res[tag] = []
        for i in range(len(text_tags)):
            if tag in text_tags[i]:
                res[tag].append(i)
    return res


if __name__ == '__main__':
    texts = []
    text_tags = []
    tags = set()
    with open('core/preprocessed_core.jsonl', encoding='utf-8') as file:
        for line in file:
            row = json.loads(line.strip())
            texts.append(row['text'])
            for i in range(len(row['tags']) - 1, -1, -1):
                if ',' in row['tags'][i]:
                    del row['tags'][i]
            text_tags.append(row['tags'])
            tags = tags.union(row['tags'])
    tag_map = group_texts_by_tags(text_tags, tags)

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

    csv_string = 'vectorizer;matrix size;matrix sparsity;semantic consistency;memory usage;computation time\n'

    for i in range(len(vectorizers)):
        for ngram_range in ngram_ranges:
            print(f'{vectorizer_names[i]} {ngram_range}')
            start_time = time.time()
            tracemalloc.start()
            vectorizer = vectorizers[i](ngram_range)
            x = vectorizer.fit_transform(texts)
            n_comps, _ = find_n_components(x, 0.999)
            svd = TruncatedSVD(n_components=n_comps, random_state=42)
            x_reduced = svd.fit_transform(x)
            _, memory_usage = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_time = time.time()

            matrix_size = x_reduced.shape #len(vectorizer.get_feature_names_out())
            matr = x.toarray()
            similarity = 0
            n_sims = 0
            n_texts = 5
            for tag in tag_map.keys():
                for ii in range(min(n_texts, len(tag_map[tag])) - 1):
                    for jj in range(ii+1, min(n_texts, len(tag_map[tag]))):
                        similarity += cosine_similarity(x_reduced[tag_map[tag][ii]].reshape(1, -1), x_reduced[tag_map[tag][jj]].reshape(1, -1))[0][0]
                        n_sims += 1
            similarity /= n_sims
            csv_string += f'{vectorizer_names[i]} {ngram_range};{matr.shape[0]}x{matr.shape[1]};{round(sparsity(matr), 3)};{round(similarity, 3)};{round(memory_usage / 1024 ** 2, 3)};{round((end_time - start_time), 3)}\n'
            print(f'{matr.shape[0]}x{matr.shape[1]}\t{round(sparsity(matr), 3)}\t{round(similarity, 3)}\t{round(memory_usage / 1024 ** 2, 3)}\t{round((end_time - start_time), 3)}')
    with open('output/vectorization_metrics.csv', 'w') as file:
        file.write(csv_string)
