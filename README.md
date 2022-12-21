# Semantic similarity

This class combines various methods to linguistically analyze and play with text. 
I use gensim, a Python library for topic modelling, document indexing and similarity retrieval with large corpora.

## My class allows to:
* **lemmatize(new_text)** – substitute each word in text with its lemma
* **modeltrain(vector_size=300, window=5, min_count=5, epochs=50)** – train model in accordance with set parameters
* **get_num_words()** – return a number of words in model created
* **get_words()** – get all the words used in a text in an alphabetical order
* **get_n_most_similar(word, n=5)** – get a number of the most similar words
* **get_semantic_proportion(positives, negatives)** – get semantic proportion for a chosen word
* **odd_one_out(line)** – in a list of words find an odd one
* **visualize_vector_similarity(words)** – visualize words similarity 
* **words_swap(input_sentence='')** – swap each word in a sentence with the most similar one

## Files description:
* **gambler.txt** – original text
* **new-text.txt** – original text lemmatized and cleaned from punctuation with each sentence on a new line
* **new-text-words.txt** – all the words listed from the new-text.txt
