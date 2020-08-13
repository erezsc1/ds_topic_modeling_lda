# LDA Topic Modeling

This Library provides a wrapper for Topic Modeling using LDA. Provdies basic visualization.
##Requirements
```requirements.txt
wordcloud==1.7.0
pandas==1.0.5
pyLDAvis==2.1.2
scikit_learn==0.23.2
```

##Example:
```python
from topic_modeling import TopicModeler
 

class MyTopicModeler(TopicModeler):
    def preprecess_data(self, sentences: pd.Series):
        # your code here

#usage
tm = MyTopicModeler(
            task="news",
            num_words=10,
            num_topics=4,
            n_jobs=8
        )

tm.fit(data_1)  # data_1 : pd.Series
tm.transform(data_2) # data_2 : pd.Series
tm.explore_topics_viz(saved_topics_path)
```

## Credit
credit goes to the following [blog post](!https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0) 