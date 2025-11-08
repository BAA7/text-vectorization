import fasttext
import streamlit as st
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os
import glob


class UnifiedVectorModel:
    def __init__(self, backend_model, model_type="w2v"):
        self.model = backend_model
        self.model_type = model_type.lower()

        if self.model_type == "w2v":
            self.wv = backend_model.wv
            self.key_to_index = self.wv.key_to_index
            self.vector_size = self.wv.vector_size
            self._words = set(self.wv.key_to_index.keys())

        elif self.model_type == "ft":
            # Для fasttext-wheel
            self.key_to_index = {word: i for i, word in enumerate(backend_model.get_words())}
            self.vector_size = backend_model.get_dimension()
            self._words = set(self.key_to_index.keys())
        else:
            raise ValueError("model_type must be 'w2v' or 'ft'")

    def __contains__(self, word):
        return word in self._words

    def __getitem__(self, word):
        if self.model_type == "w2v":
            return self.wv[word]
        elif self.model_type == "ft":
            return self.model.get_word_vector(word)

    def most_similar(self, positive=None, negative=None, topn=10):
        from sklearn.metrics.pairwise import cosine_similarity

        if not positive:
            positive = []
        if not negative:
            negative = []

        try:
            if self.model_type == "w2v":
                return self.wv.most_similar(positive=positive, negative=negative, topn=topn)

            elif self.model_type == "ft":
                vec = np.zeros(self.vector_size)
                for w in positive:
                    if w in self:
                        vec += self[w]
                    else:
                        continue
                for w in negative:
                    if w in self:
                        vec -= self[w]
                    else:
                        continue

                if np.allclose(vec, 0):
                    return []

                words = list(self._words)
                vectors = np.array([self[w] for w in words])

                sims = cosine_similarity([vec], vectors)[0]
                best = np.argsort(sims)[::-1][:topn + len(positive) + len(negative)]

                result = []
                for i in best:
                    word = words[i]
                    if word not in positive and word not in negative:
                        result.append((word, float(sims[i])))
                        if len(result) >= topn:
                            break
                return result

        except Exception as e:
            print(f"Error in most_similar: {e}")
            return []

    def similar_by_vector(self, vector, topn=10):
        from sklearn.metrics.pairwise import cosine_similarity

        words = list(self._words)
        vectors = np.array([self[w] for w in words])
        sims = cosine_similarity([vector], vectors)[0]
        best = np.argsort(sims)[::-1][:topn]

        return [(words[i], float(sims[i])) for i in best]

    def get_words(self):
        return list(self._words)

    @property
    def vectors(self):
        if not hasattr(self, '_cached_vectors'):
            words = list(self._words)
            self._cached_words = words
            self._cached_vectors = np.array([self[w] for w in words])
        return self._cached_vectors

    @property
    def index_to_key(self):
        if not hasattr(self, '_index_to_key'):
            self._index_to_key = list(self._words)
        return self._index_to_key


@st.cache_resource
def load_model(model_path):
    try:
        if model_path.endswith(".model"):
            raw_model = Word2Vec.load(model_path)
            current_model = UnifiedVectorModel(raw_model, model_type="w2v")

        elif model_path.endswith(".bin"):
            raw_model = fasttext.load_model(model_path)
            current_model = UnifiedVectorModel(raw_model, model_type="ft")
        else:
            raise ValueError(f"wrong path format")
        return current_model
    except Exception as e:
        st.error(f"error loading model {model_path}: {e}")
        return None


MODELS_DIR = "models"

if not os.path.exists(MODELS_DIR):
    st.error(f"Folder `{MODELS_DIR}` not found.")
    st.stop()

model_files = []
for ext in ["*.bin", "*.model", "*.vec"]:
    model_files.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
model_files = [f for f in model_files if os.path.isfile(f)]
model_names = [os.path.basename(f) for f in model_files]

if len(model_names) == 0:
    st.error(f"No models in folder `{MODELS_DIR}` (.bin, .model, .vec).")
    st.info("Supported formats: Word2Vec (binary/text), FastText.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Choose pretrained model",
    model_names
)

selected_model_path = os.path.join(MODELS_DIR, selected_model_name)

st.sidebar.info(f"loading: `{selected_model_name}`")

model = load_model(selected_model_path)

if model is None:
    st.stop()
else:
    st.sidebar.success(f"Model '{selected_model_name}' loaded")
    st.sidebar.write(f"Voc size: {len(model.key_to_index):,}")
    st.sidebar.write(f"Vector size: {model.vector_size}")

def analogy_accuracy(model, file_name):
    right = 0
    count = 0
    results = []
    with open(file_name, encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            if len(words) != 4:
                continue
            try:
                most_similar = model.most_similar(positive=[words[0], words[2]], negative=[words[1]], topn=10)
                predicted = [x[0] for x in most_similar]
                correct = words[3]
                if correct in predicted:
                    rank = predicted.index(correct) + 1
                    right += 1
                else:
                    rank = None
                count += 1
                results.append({
                    "query": f"{words[0]} - {words[1]} + {words[2]}",
                    "target": correct,
                    "predicted": predicted[0],
                    "rank": rank,
                    "in_top10": bool(rank)
                })
            except KeyError as e:
                continue
    accuracy = right / count if count > 0 else 0
    return accuracy, results


def avg_similarity(model, file_name):
    res = []
    with open(file_name, encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            try:
                vectors = [model[word] for word in words]
            except KeyError:
                continue
            sims = cosine_similarity(vectors)
            for i in range(len(words) - 1):
                for j in range(i + 1, len(words)):
                    res.append(sims[i][j])
    return sum(res) / len(res) if res else 0


def projection(word_vec, axis):
    axis_norm = axis / np.linalg.norm(axis)
    return np.dot(word_vec, axis_norm)


def get_projection_row(model, axis):
    words = list(model.key_to_index.keys())
    projections = [(word, projection(model[word], axis)) for word in words]
    projections = sorted(projections, key=lambda x: x[1])
    return projections


st.title("Vector embeddings")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vector ariphmetics",
    "Semantic consistency",
    "Semantic axis",
    "Distribution analysis",
    "Report"
])

with tab1:
    st.header("Vector ariphmetics")
    expr = st.text_input("Insert expression", value="рубль - россия + сша")

    if st.button("Compute"):
        words = expr.replace('+', ' + ').replace('-', ' - ').split()
        positive, negative = [], []
        current = 'pos'

        for w in words:
            if w == '+':
                current = 'pos'
            elif w == '-':
                current = 'neg'
            else:
                (positive if current == 'pos' else negative).append(w)

        missing = [w for w in positive + negative if w not in model]
        if missing:
            st.warning(f"Words not found in voc: {', '.join(missing)}")
            st.stop()

        try:
            similar = model.most_similar(
                positive=positive,
                negative=negative,
                topn=10
            )

            st.write("### Result:")
            result_words = [f"{w} ({s:.3f})" for w, s in similar]
            st.write("Nearest words: " + ", ".join(result_words))

            st.write("### In-between steps")

            cum_vec = np.zeros(model.vector_size)

            steps_data = []

            for i in range(len(positive)):
                cum_vec += model[w]
                nearest = model.most_similar(positive=positive[:i + 1], topn=1)
                steps_data.append({
                    "step": f"+ {positive[i]}",
                    "nearest word": nearest[0][0],
                    "similarity": nearest[0][1]
                })

            for i in range(len(negative)):
                cum_vec -= model[w]
                nearest = model.most_similar(positive=positive, negative=negative[:i + 1], topn=1)
                steps_data.append({
                    "step": f"- {negative[i]}",
                    "nearest word": nearest[0][0],
                    "similarity": nearest[0][1]
                })

            df_steps = pd.DataFrame(steps_data)
            st.dataframe(df_steps[["step", "nearest word", "similarity"]])

            result_word = similar[0][0]
            fig = px.scatter(
                x=[cum_vec[0]], y=[cum_vec[1]],
                text=[result_word],
                title="Result (first 2 components)"
            )
            fig.update_traces(textposition='top center', marker=dict(size=12, color='red'))
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error computing: {e}")

with tab2:
    st.header("Similarity calculator")
    col1, col2 = st.columns(2)
    with col1:
        word1 = st.text_input("word 1", value="мужчина")
    with col2:
        word2 = st.text_input("word 2", value="женщина")

    if st.button("Compute similarity"):
        try:
            v1, v2 = model[word1], model[word2]
            sim = cosine_similarity([v1], [v2])[0][0]
            st.metric("Cosine similarity", f"{sim:.4f}")

            st.write("### Nearest neighbors graph")
            neighbors = model.most_similar(word1, topn=5) + model.most_similar(word2, topn=5)
            nodes = list(set([word1, word2] + [n[0] for n in neighbors]))
            edges = [(word1, n[0]) for n in model.most_similar(word1, topn=5)] + \
                    [(word2, n[0]) for n in model.most_similar(word2, topn=5)]

            G = go.Figure()
            pos = np.random.rand(len(nodes), 2) * 2 - 1
            node_x = pos[:, 0]
            node_y = pos[:, 1]

            for edge in edges:
                x0, y0 = pos[nodes.index(edge[0])]
                x1, y1 = pos[nodes.index(edge[1])]
                G.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=1, color='gray'), showlegend=False))

            G.add_trace(go.Scatter(x=node_x, y=node_y, mode='text+markers',
                                   marker=dict(size=10, color='lightblue'),
                                   text=nodes, textposition="top center"))
            G.update_layout(title="Semantic links graph", showlegend=False)
            st.plotly_chart(G)

        except KeyError as e:
            st.error(f"Word not found: {e}")

with tab3:
    st.header("Semantic axis projection")
    col1, col2 = st.columns(2)
    with col1:
        pos_axis = st.text_input("positive", value="мужчина")
    with col2:
        neg_axis = st.text_input("negative", value="женщина")

    if st.button("Build axis"):
        try:
            pos_vec = model[pos_axis]
            neg_vec = model[neg_axis]
            axis = pos_vec - neg_vec

            projections = get_projection_row(model, axis)
            top_pos = projections[-10:][::-1]
            top_neg = projections[:10]

            st.write(f"Axis: **{pos_axis} – {neg_axis}**")
            st.write("### Top 10 positive:")
            st.write(", ".join([f"{w} ({p:.3f})" for w, p in top_pos]))

            st.write("### Top 10 negative:")
            st.write(", ".join([f"{w} ({p:.3f})" for w, p in top_neg]))

            df_proj = pd.DataFrame(top_pos + top_neg, columns=["word", "projection"])
            fig = px.bar(df_proj, x="projection", y="word", orientation='h', title=f"Projection on axis: {pos_axis}–{neg_axis}")
            st.plotly_chart(fig)

        except KeyError as e:
            st.error(f"Error: {e}")

with tab4:
    st.header("Distance distribution analysis")
    all_vectors = model.vectors
    sample = all_vectors[np.random.choice(all_vectors.shape[0], 1000, replace=False)]

    dists = cosine_similarity(sample)
    np.fill_diagonal(dists, 0)
    flat_dists = dists.flatten()
    flat_dists = flat_dists[flat_dists > 0]

    fig = px.histogram(flat_dists, nbins=50, title="Cosine similarity distribution between random words")
    st.plotly_chart(fig)

    st.metric("Mean similarity", f"{np.mean(flat_dists):.3f}")
    st.metric("Std deviation", f"{np.std(flat_dists):.3f}")

with tab5:
    st.header("Report")

    st.subheader("1. Analogy rate")
    analogies_file = "data/analogy.txt"
    if os.path.exists(analogies_file):
        acc, results = analogy_accuracy(model, analogies_file)
        st.metric("Analogy accuracy (in top 10)", f"{acc:.2%}")
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("File `analogy.txt` not found.")

    st.subheader("2. Average synonyms similarity")
    sim_file = "data/synonyms.txt"
    if os.path.exists(sim_file):
        avg_sim = avg_similarity(model, sim_file)
        st.metric("Average similarity", f"{avg_sim:.4f}")
    else:
        st.warning("File `similarity_words.txt` not found.")

    st.subheader("3. Average antonyms similarity")
    sim_file = "data/antonyms.txt"
    if os.path.exists(sim_file):
        avg_sim = avg_similarity(model, sim_file)
        st.metric("Average similarity", f"{avg_sim:.4f}")
    else:
        st.warning("File `similarity_words.txt` not found.")

    st.subheader("4. Heatmap for nearest words")
    query_words = st.text_input("Enter words", value="мужчина женщина мальчик девочка").split()
    if st.button("Build heatmap"):
        try:
            vectors = [model[w] for w in query_words]
            sims = cosine_similarity(vectors)
            fig = px.imshow(sims, x=query_words, y=query_words, color_continuous_scale="Blues", title="Similarity heatmap")
            st.plotly_chart(fig)
        except KeyError as e:
            st.error(f"Error: {e}")

    st.subheader("5. 2D projection")
    sample_words = st.text_input("Input words", value="мужчина женщина мальчик девочка")
    word_list = sample_words.split()
    if st.button("Show clusters"):
        try:
            from sklearn.manifold import TSNE
            vectors = np.array([model[w] for w in word_list])
            tsne = TSNE(n_components=2, perplexity=len(vectors) - 1, random_state=42)
            embedded = tsne.fit_transform(vectors)

            fig = px.scatter(x=embedded[:, 0], y=embedded[:, 1], text=word_list, title="words projection")
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig)
        except KeyError as e:
            st.error(f"Word not found: {e}")