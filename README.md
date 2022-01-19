# Word2Vec-Application

Word2vec is a two-layer neural net that processes text by vectorizing words. Its input is a text corpus,  and its output is a set of vectors: feature vectors that represent words in that corpus. Here the two review data is being used. Hotel Review and IMDB Movie Review data are collected from Kaggle. The data is available here https://drive.google.com/drive/folders/1D9vNJ9O11JOyXo89-Sl2bsguoSxB59nZ?usp=sharing

In IMDB movie data there are 50000 reviews, and in hotel data there are about 5 million reviews. Here we have extracted the words from the data that are contextually similar to the phrase given by the user.  

# Run the Program

In code folder the following commands need to be run to get the output and data path is needed to be set up:

python Similar_Word_extraction.py hotel   #for the hotel review data

python Similar_Word_extraction.py movie   #for the movie review data


The notebook Word2Vec_Test is available in code folder. 


Further, the Google Colab link is here: https://colab.research.google.com/drive/15aYA9vx_0T6FpeE1tEwOUyHT7CPIkBgE?usp=sharing  to run the code in Google Colab. The data sets need to be uploaded before running the code here.  

# Sample Outcome

Let's say the user input phrase is "movie", and the Word2Vec model is trained with the input text. Here the input text is the movie reviews. Word2Vec will provide the embeddings for the words in the input text. We will have also the embeddings (vector representation) for the users' input phrase using Word2Vec. Then based on the cosine similarities of the embeddings, we will get the closest words of the phrase in the text. For instance, for the "movie" phrase we get the following outcome that is the ranked list of the closest words:

['movie', 'martial', 'fight', 'original', 'special', 'establish', 'laughter', 'sick']


In case hotel review data, if the phrase is 'food', we get the ranked list of the closely related words:


['food', 'pizzas ', 'breakfast',  'steak ', 'pasta ', 'cuisine ', 'bananas ', 'desserts ', 'hams ', 'pastries ']


More examples of output:


Movie Review Data:

Phrase:godzilla


Extracted Similar Word List: ['reese', 'forgive', 'terrible', 'hollywood', 'dominic', 'mean', 'party', 'watershed', 'shaw', 'bart']


Phrase:gangster


Extracted Similar Word List: ['killer', 'kristina', 'ozjeppe', 'country', 'hollywood', 'heart', 'cinematographer', 'forgive', 'comedy', 'masterpiece']


Phrase:zombie


Extracted Similar Word List: ['terrible', 'party', 'masterpiece', 'country', 'cinematographer', 'hollywood', 'watershed', 'ozjeppe', 'heart', 'forgive']


Phrase: murder


Extracted Similar Word List: ['killer', 'shaw', 'terrible', 'right', 'musical', 'ozjeppe', 'heart', 'bart', 'hollywood', 'flick']


Hotel Review Data:

Phrase: dinner


Extracted Similar Word List: ['dinner', 'average', 'buffet', 'shop', 'space', 'breakfast', 'station', 'quirky', 'level', 'good']


Phrase: restaurant


Extracted Similar Word List: ['nice', 'hassle', 'good', 'different', 'issue', 'location', 'lounge', 'excellent', 'great', 'link']


Phrase:cafe



Extracted Similar Word List: ['average', 'buffet', 'view', 'design', 'step', 'amaze', 'wifi', 'overall', 'decor', 'clean']



