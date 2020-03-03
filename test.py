from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

model= Doc2Vec.load("en-wiki-03032020-small.model")

similar_doc = model.docvecs.most_similar(positive=[model.infer_vector("Embassy".split())], topn=20)
print(similar_doc)
