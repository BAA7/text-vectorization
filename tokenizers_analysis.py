import json
from nltk.tokenize import word_tokenize
from razdel import tokenize as razdel_tokenize
from snowballstemmer import RussianStemmer
import pymorphy2
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_naive_space(text):
    return text.split(' ')


def tokenize_regex(text):
    return re.findall(r'[а-яА-ЯёЁ]+', text)


def tokenize_nltk(text):
    return word_tokenize(text, language='russian')


def tokenize_razdel(text):
    return list(map(lambda x: x.text, razdel_tokenize(text)))


def stem_snowball(tokens):
    stemmer = RussianStemmer()
    return stemmer.stemWords(tokens)


def lemmatize_pymorphy(words):
    morph = pymorphy2.MorphAnalyzer()
    lemmas = []
    for word in words:
        lemmas.append(morph.parse(word)[0].normal_form)
    return lemmas


def calculate_oov(text, vocabulary):
    words = text.split(' ')
    oov_count = 0
    for word in words:
        if word not in vocabulary:
            oov_count += 1
    return oov_count / len(words)


def calculate_similarity(text1, text2, model):
    embedding1 = model.encode(text1, convert_to_tensor=False).reshape(1, -1)
    embedding2 = model.encode(text2, convert_to_tensor=False).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


if __name__ == '__main__':
    texts = []
    with open('core/preprocessed_core.jsonl', encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            texts.append(row['text'])
    n_articles = len(texts)
    tokenizers = [tokenize_naive_space, tokenize_regex, tokenize_nltk, tokenize_razdel]

    methods = [[tokenize_razdel, lemmatize_pymorphy]]
    # for tokenizer in tokenizers:
    #     methods.extend([[tokenizer], [tokenizer, stem_snowball], [tokenizer, lemmatize_pymorphy]])

    csv_string = 'method;vocabulary volume;OOV percentage;processing speed;semantic consistency\n'
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    for method in methods:
        print('running', " + ".join(map(lambda x: x.__name__, method)))
        start_time = time.time()
        vocabulary = set()
        similarities = []
        for text in texts:
            tokens = text
            for func in method:
                tokens = func(tokens)
            similarities.append(calculate_similarity(text, ' '.join(tokens), sim_model))
            with open(f'tokenized_texts/{"_".join(map(lambda x: x.__name__, method))}', 'a', encoding='utf-8') as file:
                file.write(' '.join(tokens) + '\n')
            vocabulary = vocabulary.union(tokens)
        end_time = time.time()
        csv_string += f'{" + ".join(map(lambda x: x.__name__, method))};{len(vocabulary)};'
        csv_string += f'{calculate_oov(" ".join(texts), vocabulary)};{end_time - start_time};'
        csv_string += f'{sum(similarities)/len(similarities)}\n'
    # with open('reports/tokenization_metrics.csv', 'w') as file:
    #     file.write(csv_string)
