import os
import json
import numpy as np
from gensim.models import Word2Vec
import fasttext

from vectorizers_analysis import group_texts_by_tags


if __name__ == '__main__':
    texts = []
    text_tags = []
    tags = set()
    with open('core/preprocessed_core.jsonl', encoding='utf-8') as file:
        for line in file:
            row = json.loads(line.strip())
            for i in range(len(row['tags']) - 1, -1, -1):
                if ',' in row['tags'][i]:
                    del row['tags'][i]
            text_tags.append(row['tags'])
            tags = tags.union(row['tags'])
    with open('core/core_tokenized', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip().split())
    tag_map = group_texts_by_tags(text_tags, tags)

    vector_sizes = [100, 200, 300]
    windows = [5, 8, 10]
    min_counts = [5, 8, 10]
    n_epochs = 100
    for vs in vector_sizes:
        for w in windows:
            for count in min_counts:
                files = os.listdir('models')
                if f'w2v_v{vs}_w{w}_m{count}_sg0.model' not in files:
                    print(f'adding w2v_v{vs}_w{w}_m{count}_sg0.model')
                    model = Word2Vec(sentences=texts,
                                     vector_size=vs,
                                     window=w,
                                     min_count=count,
                                     workers=4,
                                     sg=0,
                                     epochs=n_epochs)
                    model.save(f'models/w2v_v{vs}_w{w}_m{count}_sg0.model')
                if f'w2v_v{vs}_w{w}_m{count}_sg1.model' not in files:
                    print(f'adding w2v_v{vs}_w{w}_m{count}_sg1.model')
                    model = Word2Vec(sentences=texts,
                                     vector_size=vs,
                                     window=w,
                                     min_count=count,
                                     workers=4,
                                     sg=1,
                                     epochs=n_epochs)
                    model.save(f'models/w2v_v{vs}_w{w}_m{count}_sg1.model')
                if f'ft_v{vs}_w{w}_m{count}_sg.bin' not in files:
                    print(f'adding ft_v{vs}_w{w}_m{count}_sg.bin')
                    model = fasttext.train_unsupervised(
                        'core/core_tokenized',
                        model='skipgram',
                        dim=vs,
                        epoch=n_epochs,
                        minCount=count,
                        minn=1,
                        maxn=3
                    )
                    model.save_model(f'models/ft_v{vs}_w{w}_m{count}_sg.bin')
                if f'ft_v{vs}_w{w}_m{count}_cbow.bin' not in files:
                    print(f'adding ft_v{vs}_w{w}_m{count}_cbow.bin')
                    model = fasttext.train_unsupervised(
                        'core/core_tokenized',
                        model='cbow',
                        dim=vs,
                        epoch=n_epochs,
                        minCount=count,
                        minn=1,
                        maxn=3
                    )
                    model.save_model(f'models/ft_v{vs}_w{w}_m{count}_cbow.bin')
