from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import json


def find_n_components(x, variance_threshold=0.95, get_variance_array=False):
    max_comp = min(1000, x.shape[0], x.shape[1])

    svd = TruncatedSVD(n_components=max_comp, random_state=42)
    svd.fit(x)

    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    if get_variance_array:
        return n_components, cumulative_variance
    return n_components, cumulative_variance[n_components - 1]


if __name__ == '__main__':
    texts = []
    with open('core/core_tokenized', encoding='utf-8') as file:
        texts = file.readlines()

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)

    n_comps, variance = find_n_components(x, 0.999, get_variance_array=True)
    # plt.figure()
    # plt.plot(range(1, len(variance) + 1), variance)
    # plt.xlabel('n_components')
    # plt.ylabel('variance')
    # plt.show()

    print(n_comps)

    svd = TruncatedSVD(n_components=n_comps, random_state=42)
    x_reduced = svd.fit_transform(x)

    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=1,
        n_components=2
    )
    x_umap = reducer.fit_transform(x_reduced)

    df = pd.DataFrame({
        'x': x_umap[:, 0],
        'y': x_umap[:, 1],
        'text': [t.strip()[:50] for t in texts]
    })

    fig = px.scatter(
        df, x='x', y='y',
        hover_data={'text': True},
        # title='Визуализация текстов (UMAP)',
        width=900, height=700
    )
    fig.show()
