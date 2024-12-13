from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd
import gensim
import nltk
from pathlib import Path
import pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

script_path = Path(__file__).resolve()
transformer_directory = script_path.parent.parent.parent.parent / "savedModels" / "transformer"
print(transformer_directory)
most_recent_transformer = max(transformer_directory.glob("*"), key=lambda f: f.stat().st_mtime)

def load_transformer():
    print("Most Recent Model:", most_recent_transformer)
    try:
        transformer = pickle.load(open(most_recent_transformer, 'rb'))
        return transformer
    except FileNotFoundError:
        print("The pickle file does not exist.")
    except pickle.UnpicklingError:
        print("The pickle file could not be loaded.")

# Function to compute sentence embeddings
def sentence_to_vector(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def extract(df):
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in df['reviewText']]
    word2vec_model = gensim.models.Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=1)
    print(word2vec_model)
    # Generate Word2Vec sentence embeddings
    word2vec_features = np.array([sentence_to_vector(sentence, word2vec_model) for sentence in tokenized_sentences])

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 3), stop_words='english')

    # Generate CountVectorizer features
    Tfidf_vectorizer_features = vectorizer.fit_transform(df['reviewText']).toarray()

    # Combine Word2Vec and CountVectorizer features
    combined_features = np.hstack((word2vec_features, Tfidf_vectorizer_features))

    df = pd.DataFrame({
        'reviewText': df['reviewText'],
        'combined_features': list(combined_features),  # Store combined features as lists
        'sentiment': df['sentiment']
    })
    return df
