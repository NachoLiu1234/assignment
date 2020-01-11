from flask import Flask
from flask import request, render_template
from gensim import models
from gensim.models.word2vec import LineSentence, Word2Vec
import jieba, re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import json


model = Word2Vec.load('word2vec.model')
vlookup = model.wv.vocab  # Gives us access to word index and count
Z = 0
for k in vlookup:
    Z += vlookup[k].count  # Compute the normalization constant Z

def get_sentenses_vector(sentences, model=model, alpha=1e-3, Z=Z):
    def sif_embeddings(sentences, model, alpha=alpha, Z=Z):
        vlookup = model.wv.vocab  # Gives us access to word index and count
        vectors = model.wv  # Gives us access to word vectors
        size = model.vector_size  # Embedding size
        output = []

        # Iterate all sentences
        for s in sentences:
            count = 0
            v = np.zeros(size, dtype=np.float32)  # Summary vector
            # Iterare all words
            for w in s:
                # A word must be present in the vocabulary
                if w in vlookup:
                    v += (alpha / (alpha + (vlookup[w].count / Z))) * vectors[w]
                    count += 1
            if count > 0:
                v /= count
            output.append(v)
        return np.vstack(output).astype(np.float32)
    vector = sif_embeddings(sentences, model)
    pca = PCA(1)
    pca.fit(vector)
    u = pca.components_[0]
    #     for i in range(len(vector)):
    #         vector[i] -= np.multiply(np.multiply(u, u.T), vector[i])
    vector -= np.multiply(np.multiply(u, u.T), vector)
    return vector

sentence_pattern = re.compile('[！？。…\r\n\\n\\r]+')
def get_all_list(content, title):
    if not (type(content) == str and type(title) == str):
        return (None, None, None)
    sentense = [el.replace('\\r', '').replace('\\n', '') for el in sentence_pattern.split(content)]
    sentense = [jieba.lcut(el) for el in sentense if el]
    title = jieba.lcut(title)
    content = jieba.lcut(content)
    return content, title, sentense

def calculate_knn_value(v_list):
    content_list = [el[0] for el in v_list]
    value_list = np.array([el[1] for el in v_list])
    value_start = (value_list[0] + value_list[1]) / 2
    value_end = (value_list[-1] + value_list[-2]) / 2
    value_list = (value_list[1: -1] + value_list[:-2] + value_list[2:]) / 3
    value_list = np.append(value_start, value_list)
    value_list = np.append(value_list, value_end)
    v_list = list(zip(content_list, value_list))
    return v_list


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/abstract', methods=['POST'])
def abstract_title_content():
    title = request.form['title']
    content = request.form['content']
    content, title, sentense = get_all_list(content, title)
    if not content or not title or not sentense:
        return json.dumps('invalid')
    v_content = get_sentenses_vector([content])
    v_title = get_sentenses_vector([title])
    v_sentense = get_sentenses_vector(sentense)
    v_target = (v_title + v_content) / 2

    v_list = [(''.join(s), cosine(v, v_target)) for s, v in zip(sentense, v_sentense)]
    v_list = [el for el in v_list if not np.isnan(el[1])]
    knn_list = calculate_knn_value(v_list)
    knn_list.sort(key=lambda x: x[1])
    return json.dumps(knn_list[:3])

if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)
