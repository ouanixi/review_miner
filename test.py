from app.mltrainers.dataloaders import InfLoader, Word2vecLoader


loader = InfLoader()

loader.vectoriser = loader.get_vector
word2vec = Word2vecLoader().get_model

word_centroid_map = dict(zip(word2vec.wv.index2word,
                             loader.vectoriser.predict(word2vec.wv.syn0)))


# for cluster in range(210, 220):
#     #
#     # Print the cluster number
#     print("\nCluster %d" % cluster)
#     #
#     # Find all of the words for that cluster number, and print them out
#     words = []
#     for i in range(0, len(word_centroid_map.values())):
#         if(list(word_centroid_map.values())[i] == cluster):
#             words.append(list(word_centroid_map.keys())[i])
#     print(words)
