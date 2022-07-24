import pandas as pd
from tqdm.autonotebook import tqdm


class GenericRecommender:

    def __init__(self, dataset_metadata, model, top_n: int = 10):
        self.top_n = top_n
        self.dataset = dataset_metadata.dataset
        self.model = model
        self.catalogue = set(self.dataset['itemId'])

    def recommend_all(self):
        """
        Get all recommendations.
        :param top_n:
        :return: recommendations for any user.
        """

        ratings = self.dataset.groupby('userId')
        
        #made an empty recommendations list

        recommendations = pd.DataFrame({'userId': [], 'itemId': [], 'rank': []})
        
        # el heta el fl awel el abl append di bet3mel shakl el progress bar el fl notebook then it appends the recommendations obtained by calling recommend_user() funtion which is in recommender.py

        with tqdm(total=self.dataset['userId'].nunique(), desc="Recommending for users: ") as pbar:
            for user_id, user_ratings in ratings:
                recommendations = recommendations \
                    .append(self.recommend_user(user_id, user_ratings))
                pbar.update()

        return recommendations
    
    #badkhol b user id w target item id. predictions di gaya men en recommend bet call get predicitions fa betraga3ha le recommend fa recommend bet call rank predicitions bel predicitions el gatlaha di. apparently howa bey rank el recommendation descendingly based 3al prediction wyegeeb el top N hasab ana mehadedaha ehh

    def rank_prediction(self, user_id, target_item_id, predictions):
        recommendations = pd.DataFrame({'userId': user_id,
                                        'itemId': target_item_id,
                                        'prediction': predictions})

        recommendations['rank'] = recommendations['prediction'] \
            .rank(method='first', ascending=False)

        recommendations \
            .sort_values(['userId', 'rank'], inplace=True)

        recommendations = recommendations[recommendations['rank'] <= self.top_n]

        return recommendations[['userId', 'itemId', 'rank']]

    def get_unrated(self, user_ratings):
        """
        Extract the set of items a user has not rated.
        :param user_ratings: list, items rated.
        :return: list, items not rated.
        """
        unrated_item_id = self.catalogue - set(user_ratings)
        unrated_item_id = list(unrated_item_id)
        return unrated_item_id

    def get_rated(self, user_id):
        """
        Extract the set of items a user has not rated.
        :param user_id: userId rated.
        :return: list, rated items.
        """
        rated = self.dataset[self.dataset['userId'] == user_id]
        return rated
