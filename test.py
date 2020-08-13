import os
import pickle
import pandas as pd
from topic_modeling import TopicModeler


if __name__ == '__main__':
    data_path = os.path.join("datasets", "ag_news.csv")
    tm_path = "topic_model.pkl"
    saved_topics_path = os.path.join("lda_vis_postprocessed")
    data = pd.read_csv(data_path)

    if not os.path.exists(tm_path):
        tm = TopicModeler(
            task="news",
            num_words=10,
            num_topics=4,
            n_jobs=8
        )
        tm.fit(data)
        tm.explore_topics_viz(saved_topics_path)

        with open(tm_path, "wb") as fp:
            pickle.dump(tm, fp)
    else:
        with open(tm_path, "rb") as fp:
            tm = pickle.load(fp)

    test_sentences = pd.Series(
        [
            "Maccabi Haifa has won the israeli football league for the third time in a row",
            "Israel bombs iranian nuclear silo",
            "Hapoel Tel-Aviv has won the israeli football league for the third time in a row",
            "After long negotiations, Microsoft accquires google"
         ]
    )
    print(tm.predict_topic(test_sentences))


