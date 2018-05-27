import sent2vec
from scipy import spatial
model = sent2vec.Sent2vecModel()
model.load_model('wiki_bigrams.bin')
emb = model.embed_sentence("once upon a time .") 
embs = model.embed_sentences(["first sentence .", "another sentence"]) 
emb1 = model.embed_sentence("This is lsma")
emb2 =  model.embed_sentence("This is lsma2")
result = 1 - spatial.distance.cosine(emb1, emb2)
print (result)
