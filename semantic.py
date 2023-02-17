#Compulsory Task 1

import spacy
nlp = spacy.load("en_core_web_md")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

#The most similar words are cat and monkey, presumably because they are both animals.
#It's interesting that monkey and banana are more similar than cat and banana, so there is
#a link made between not just what each word is; animal or food, but that certain animals 
#eat certain foods, making those words more similar.
 
tokens = nlp("cat apple monkey banana")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
        
#playing around with the code to show interesting relationships/think of an example of your own
tokens = nlp("cat mouse cheese sausages")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

tokens = nlp("dog wolf cat monkey")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

tokens = nlp("cat wolf dog monkey")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#running example file with en_core_web_sm resulted in the following warning:

#The model you're using has no word vectors loaded, so the result of the Doc.similarity method 
# will be based on the tagger, parser and NER, which may not give useful similarity judgements. 
# This may happen if you're using one of the small models, e.g. 
#`en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your 
#own word vectors, or use one of the larger models instead if available.

#The similarity was not as accurate. For example the scores between the sentences related to mortages in the 
#complaints section did not score as closely as they did with the md model