from gensim import downloader
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors
from flask import Flask, request, jsonify
import time
from Quantisation import Quantisation
from config import *
import sys
import gensim.downloader as api

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Welcome to the API for comparing the results of ANN and KNN algos!'

@app.route('/ann', methods = ['GET', 'POST'])
def ANN():
    start = time.time()

    query = request.args.get('query', type = str)
    top_k = request.args.get('top_k', type = int)

    try:
        query_vector = w2v_model.wv[query]
        nearest_neighbors = my_quantisation_object.get_approximate_nearest_neighbors(query=query_vector, top_k= top_k)
        nearest_neighbors = [w2v_model.wv.index_to_key[id] for id in nearest_neighbors]
        
    except KeyError:
        print(f"The word '{query}' does not exit !")
        nearest_neighbors = []

    return jsonify({
        'latency': time.time() - start,
        'approximate_nearest_neighbors': nearest_neighbors,
        'status': '200 ok'
    })


@app.route('/knn', methods = ['GET', 'POST'])
def KNN():
    start = time.time()

    query = request.args.get('query', type = str)
    top_k = request.args.get('top_k', type = int)

    try:
        nearest_neighbors = [token for (token, _) in w2v_model.wv.most_similar(query, topn= top_k)]
        # print(nearest_neighbors)
    except KeyError:
        print(f"The word '{query}' does not exit !")
        nearest_neighbors = []

    return jsonify({
        'latency': time.time() - start,
        'nearest_neighboers': nearest_neighbors,
        'status': '200 ok'
    })



if __name__ == '__main__':

    # print(api.load("20-newsgroups", return_path=True)) 
    corpus = api.load('text8')  # download the corpus and return it opened as an iterable
    w2v_model = Word2Vec(corpus)  # train a model from the corpus
    
    print(f'Dimension of each vector: {w2v_model.vector_size}')
    print(f'Number of vectors : {len(w2v_model.wv.key_to_index)}')

    Input_vector_set = w2v_model.wv.vectors

    print(f"The size consumed the vectors : {sys.getsizeof(Input_vector_set)}")

    my_quantisation_object = Quantisation(Input_vector_set, compression_factor, number_of_clusters)

    print(f"The size consumed by the quantised vectors : {sys.getsizeof(my_quantisation_object.quantised_vectors)}")

    app.run(debug=True, host='0.0.0.0', port=5000)
