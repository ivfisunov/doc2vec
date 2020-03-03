from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
from datetime import datetime
import multiprocessing

lastTime = datetime.now()

print('1. Loading wiki dump...', 'Time:', datetime.now())

wiki = WikiCorpus("enwiki-small.bz2")

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
        self.strLength = 0

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            print(' ' * self.strLength, end='\r')
            print(title, end='\r')
            self.strLength = len(title) + 2
            yield TaggedDocument(words=[c for c in content], tags=[title])

documents = TaggedWikiDocument(wiki)

print('Wiki dump is loaded.', 'Time:', datetime.now(),
      'Duration: ', datetime.now() - lastTime)

cores = multiprocessing.cpu_count()
print('The number of CPUs:', cores)

models = [
    # PV-DBOW
    Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8,
            min_count=19, epochs=10, workers=cores-2),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8,
            min_count=19, epochs=10, workers=cores-2),
    # Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8, alpha=0.025,
    #         min_alpha=0.00025,
    #         min_count=5, epochs=10, workers=cores-2),
]

print('2. Start building vocab...', 'Time:', datetime.now())
lastTime = datetime.now()
# models[0].build_vocab(documents)
models[0].build_vocab(documents)
print(str(models[0]))
models[1].reset_from(models[0])
print(str(models[1]))
print('Vocabulary is built:', 'Time:', datetime.now(),
      'Duration: ', datetime.now() - lastTime)

print('3. Training is started! Take a BIG cup of coffee...', 'Time:', datetime.now())
lastTime = datetime.now()
for model in models:
    model.train(documents,
                total_examples=model.corpus_count,
                epochs=model.epochs)
print('Training is finished!', 'Time:', datetime.now(),
      'Duration: ', datetime.now() - lastTime)

print('4. Saving model...', datetime.now())
lastTime = datetime.now()
model.save("en-wiki-03032020-small.model")
print("Model Saved !!!!!!!!!!!")
print('Wow!!! That is all!', datetime.now(),
      'Duration: ', datetime.now() - lastTime)

# -----------------------------

for model in models:
    print(str(model))
    pprint(model.docvecs.most_similar(positive=[model.infer_vector("Walt Disney Pictures".split())], topn=10))

for model in models:
    print(str(model))
    pprint(model.docvecs.most_similar(positive=[model.infer_vector("Russian Federation".split())], topn=10))
