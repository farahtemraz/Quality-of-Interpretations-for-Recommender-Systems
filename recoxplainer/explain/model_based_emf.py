from tqdm.auto import tqdm

from .explainer import Explainer


class EMFExplainer(Explainer):
    def __init__(self, model, recommendations, data):
        super(EMFExplainer, self).__init__(model, recommendations, data)

    def explain_recommendation_to_user(self, user_id: int, item_id: int):
        """
        Measuring the contribution of each item to the recommendation.
        :param user_id:
        :param item_id: recommendation
        :return: returns a dataframe with the contribution to the recommendation of each previously interacted with item.
        """

        ratings_on_item = self.dataset[self.dataset.itemId == item_id] # bageeb el rows el fl dataset el feha el item el recommended lel user da
        similar_users = self.model.sim_users[user_id]
        similar_users_ratings_on_item = ratings_on_item[
            ratings_on_item.userId.isin(similar_users)
        ]
        
        # the above part gets similar users to the user that we are working on and get their ratings on the item that got recommended to our user

        explanation_df = similar_users_ratings_on_item.groupby(by='rating').count() # 1star -> 5 user, etc.
        explanation = {}

        for index, row in explanation_df.iterrows():
            explanation[index] = row[0]

        return explanation
