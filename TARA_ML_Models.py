import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Load and preprocess data
wisata_place = pd.read_csv(r'D:\TARA_dataset.csv')
wisata_place = wisata_place[wisata_place['City'] == 'Yogyakarta']

stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words()

corpus = wisata_place['Description'].astype(str)
wisata_place['Description'] = corpus.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, max_df=0.8, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(wisata_place['Description'])
terms = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)


class ContentBasedFilteringModel:
    def preprocess_text(text, stopwords):
        cleaned_text = ' '.join([word for word in text.split() if word.lower() not in stopwords])
        return cleaned_text

    def get_user_preferences(user_input):
        user_preferences = f'{user_input}'
        return user_preferences

    def get_recommendations(user_preferences, tfidf_vectorizer, df_tfidf, item_data):
        user_preferences = ContentBasedFilteringModel.preprocess_text(user_preferences, tfidf_vectorizer.get_stop_words())
        user_vector = tfidf_vectorizer.transform([user_preferences]).toarray()

        tfidf_similarities = cosine_similarity(user_vector, df_tfidf)
        tfidf_similar_items = tfidf_similarities.argsort()[0][::-1]
        # Get recommendations based on TF-IDF
        tfidf_recommendations = item_data.iloc[tfidf_similar_items[:15]]

        return tfidf_recommendations

class TFFilteringModel(ContentBasedFilteringModel):
    def build_model(self, df_tfidf_columns): 
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(df_tfidf_columns,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error') 
        return model

    def train_model(self, model, X_train, y_train, epochs, saved_model_path='TARA_MODEL'):
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=8)

        # Evaluate the model on the training data
        y_predict = model.predict(X_train).flatten()
        mse = mean_squared_error(y_train, y_predict)
         # Save model using SavedModel format
        tf.saved_model.save(model, saved_model_path)

        # Save model in h5 format
        model.save('TARA_MODEL.h5')


tf_model = TFFilteringModel()
tf_model.train_model(
    tf_model.build_model(len(tfidf_df.columns)),
    tfidf_df.values,
    wisata_place['Rating'].values,
    epochs=50
)
user_input = "tempat bermain, bukit" # if you want to test it
user_preferences = ContentBasedFilteringModel.get_user_preferences(user_input)

# Get recommendations
recommendations = ContentBasedFilteringModel.get_recommendations(user_preferences, tfidf_vectorizer, tfidf_df, wisata_place)

# Display recommendations
print(recommendations)