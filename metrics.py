from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np


def analogy_accuracy(model, file_name):
    right = 0
    count = 0
    with open(file_name, encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            if model.wv.most_similar(positive=[words[0], words[2]], negative=[words[1]], topn=3) == words[3]:
                right += 1
            count += 1
    return right / count


def avg_similarity(model, file_name):
    res = []
    with open(file_name, encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            try:
                vectors = list(map(lambda x: model.wv[x], words))
            except KeyError:
                continue
            sims = cosine_similarity(vectors)
            for i in range(len(words) - 1):
                for j in range(i + 1, len(words)):
                    res.append((sims[i][j] + sims[j][i]) / 2)
    return sum(res) / len(res)


def projection(word, axis):
    return np.dot(word, axis / np.linalg.norm(axis))


def get_projection_row(model, axis):
    words = model.wv.key_to_index
    words = list(map(lambda x: (x, projection(model.wv[x], axis)), words))
    words = sorted(words, key=lambda x: x[1])
    return words


if __name__ == '__main__':
    vector_sizes = [100, 200, 300]
    windows = [5, 8, 10]
    min_freqs = [5, 8, 10]
    for v_size in vector_sizes:
        for window in windows:
            for freq in min_freqs:
                for sg in [0, 1]:
                    model_name = f'models/w2v_v{v_size}_w{window}_m{freq}_sg{sg}'
                    model = Word2Vec.load(f'{model_name}.model')
                    print(model_name)
                    print('synonyms', avg_similarity(model, 'data/synonyms.txt'))
                    print('antonyms', avg_similarity(model, 'data/antonyms.txt'))
                    print('analogies', analogy_accuracy(model, 'data/analogy.txt'))
                    print('axis:')
                    with open('data/axis.txt', encoding='utf-8') as file:
                        for line in file:
                            words = line.strip().split()
                            axis = model.wv[words[1]] - model.wv[words[0]]
                            pr_row = get_projection_row(model, axis)
                            print(f'{words[0]}-{words[1]}: {pr_row[:5]} ... {pr_row[-5:]}')
                    print('nearest neighbors:')
                    with open('data/nearest_neighbors.txt', encoding='utf-8') as file:
                        for line in file:
                            word = line.strip()
                            print(f'{word}: {model.wv.most_similar([word], topn=10)}')
