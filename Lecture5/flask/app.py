from flask import Flask
from flask import request, render_template

##################################################### 项目一  ####################################################

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

##################################################### 项目二  ####################################################

import tensorflow as tf
import pandas as pd
import numpy as np
import gensim, time
import pickle, json
import jieba

word2vec_path = './cn.cbow.bin'
model_path = './models/model_%s.tf'
stop_word_path = '百度停用词表.txt'
truncate = 450
dimension = 300

print(1)

with open(stop_word_path, 'r', -1, 'utf8') as f:
    stop_words = f.readlines()
stop_words = set([el.rstrip('\n') for el in stop_words if el.rstrip('\n')])

to_categorical = {
    -2: np.array([1., 0., 0., 0.], dtype=np.float32),
    -1: np.array([0., 1., 0., 0.], dtype=np.float32),
    0: np.array([0., 0., 1., 0.], dtype=np.float32),
    1: np.array([0., 0., 0., 1.], dtype=np.float32),
}

word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, unicode_errors='ignore')
wv = word2vec.wv
del word2vec

columns = ['location_traffic_convenience',
   'location_distance_from_business_district', 'location_easy_to_find',
   'service_wait_time', 'service_waiters_attitude',
   'service_parking_convenience', 'service_serving_speed', 'price_level',
   'price_cost_effective', 'price_discount', 'environment_decoration',
   'environment_noise', 'environment_space', 'environment_cleaness',
   'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
   'others_overall_experience', 'others_willing_to_consume_again']

model_dict = {key: tf.keras.models.load_model(model_path % key) for key in columns}

print(2)

def get_predict_class(data, model, batch_size=1):  #(1, 450, 300)
    def get_class(numpy_result):
        result = list(numpy_result).index(max(numpy_result)) - 2
        return result

    # batch_size = min(batch_size, len(data))
    data = data[:batch_size]
    batch_result = model.predict(data, batch_size=batch_size)

    predict_truth_class = []
    for i in range(batch_size):
        predict_truth_class.append(get_class(batch_result[i]))
    return predict_truth_class
##################################################### 项目三  ####################################################


print(3)


app = Flask(__name__)

##################################################### 项目一  ####################################################

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

##################################################### 项目二  ####################################################

@app.route('/waimai')
def waimai():
    return render_template("waimai.html")

@app.route('/waimai_predict', methods=['POST'])
def waimai_api():

    print(4)
    content = request.form['content']
    content = jieba.lcut(content)
    content = [el for el in content if el not in stop_words]
    content = [wv[el] for el in content if el in wv][:truncate]  # (65, 300)

    print(5)
    add = [np.zeros(300, dtype=np.float32)]
    if len(content) < truncate:
        content.extend(add * (truncate - len(content)))
    content = np.array(content).reshape(1, truncate, dimension)

    print(6)
    result = {'info': '-2 未提及, -1 不好, 0 中性, 1 好'}
    for key, model in model_dict.items():
        predict = get_predict_class(content, model)[0]  # -2
        result[key] = predict

    print(7)
    return json.dumps(result)

##################################################### 项目三  ####################################################


if __name__ == '__main__':
    app.run('0.0.0.0', 6006, debug=True)
