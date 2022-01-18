# Word2Vec-Application

Word2vec is a two-layer neural net that processes text by vectorizing words. Its input is a text corpus,  and its output is a set of vectors: feature vectors that represent words in that corpus. Here the two review data is being used. Hotel Review and IMDB Movie Review data are collected from Kaggle. The data is available here https://drive.google.com/drive/folders/1D9vNJ9O11JOyXo89-Sl2bsguoSxB59nZ?usp=sharing

In IMDB movie data there are 50000 reviews, and in hotel data there are about 5 million reviews. Here we have extracted the words from the data that are contextually similar to the phrase given by the user.  

# Run the Program

In code folder the following commands need to be run to get the output and data path is needed to be set up:

python Similar_Word_extraction.py hotel   #for the hotel review data

python Similar_Word_extraction.py movie   #for the movie review data


The notebook Word2Vec_Test is available in code folder. 

# Sample Outcome

Let's say the user input phrase is "movie", and the Word2Vec model is trained with the input text. Here the input text is the movie reviews. Word2Vec will provide the embeddings for the words in the input text. We will have also the embeddings (vector representation) for the users' input phrase using Word2Vec. Then based on the cosine similarities of the embeddings, we will get the closest words of the phrase in the text. For instance, for the "movie" phrase we get the following outcomes that are the ranked list of the closest words:

['movie', 'martial', 'fight', 'original', 'special', 'nothing', 'smile', 'make', 'precictable', 'hail', 'inch', 'establish', 'laughter', 'sick', 'instead', 'future', 'vision', 'crazy', 'surprisingly', 'wrench']
