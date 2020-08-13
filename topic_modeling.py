import re
import os
import pickle
import pyLDAvis
import pandas as pd

from wordcloud import WordCloud
from pyLDAvis import sklearn as sklearn_lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


def print_topics(model, count_vectorizer, n_top_words):
    topics = {}
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        topic_words = " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics[topic_idx] = topic_words
        print("\nTopic #%d:" % topic_idx)
        print(topic_words)
    return topics


class TopicModeler():
    '''
    Inherit and override preprocess_data to desired needs
    '''
    def __init__(
            self,
            task,
            lang="english",
            stop_words_path=None,
            random_state=42,
            num_topics=5,
            num_words=5,
            n_jobs=-1):
        if stop_words_path is not None:
            if os.path.exists(stop_words_path) and lang is not "english":
                self.stop_words = pd.read_csv(stop_words_path).tolist()
        else:
            self.stop_words = "english"
        self.count_vectorizer = CountVectorizer(stop_words=self.stop_words)
        self.num_topics = num_topics
        self.num_words = num_words
        self.n_jobs = n_jobs
        self.task = task
        self.random_state = random_state
        self.topics = {}

    def preprocess_data(self, sentences: pd.Series):
        '''
        :param sentences pd.Series:
        :return: pd.Series
        '''
        regex1 = "[,\.!?]"
        regex2 = "<[^>]*>"
        preprocessed_data = sentences.apply(lambda x: re.sub( regex1 , '', x))
        preprocessed_data = preprocessed_data.apply(lambda x: re.sub(regex2, '', x))
        print(preprocessed_data)
        return preprocessed_data


    def word_cloud_data(self, sentences: pd.Series):
        long_string = ','.join(list(sentences.values))
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        wordcloud.generate(long_string)
        wordcloud.to_file("wordloud.png")


    def explore_topics_viz(self, save_path):
        self.LDAvis_prepared = sklearn_lda.prepare(self.lda, self.count_data, self.count_vectorizer)
        LDAvis_data_filepath = os.path.join(f"{save_path}_{self.task}_{self.num_topics}")
        with open(LDAvis_data_filepath + ".pkl", "wb") as fp:
            pickle.dump(self.LDAvis_prepared, fp)
            pyLDAvis.save_html(self.LDAvis_prepared, LDAvis_data_filepath + '.html')

    def fit(self, sentences: pd.Series):
        sentences = self.preprocess_data(sentences)
        self.count_data = self.count_vectorizer.fit_transform(sentences)
        self.lda = LDA(
            n_components=self.num_topics,
            n_jobs=-1,
            random_state=self.random_state
        )

        self.lda.fit(self.count_data)

        # Print the topics found by the LDA model
        print("Topics found via LDA:")
        self.topics = print_topics(
            self.lda,
            self.count_vectorizer,
            self.num_words
        )

    def get_topics(self):
        return self.topics

    def predict_topic(self, sentences):
        data = self.preprocess_data(sentences)
        count_vec = self.count_vectorizer.transform(data)
        topic_proba = self.lda.transform(count_vec)
        return topic_proba



