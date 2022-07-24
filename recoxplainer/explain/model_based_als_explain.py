import numpy as np
import pandas as pd

from .explainer import Explainer


class ALSExplainer(Explainer):
    def __init__(self, model, recommendations, data, number_of_contributions=10):
        super(ALSExplainer, self).__init__(model, recommendations, data)
        self.number_of_contributions = number_of_contributions

    def explain_recommendation_to_user(self, user_id: int, item_id: int):
        """
        Measuring the contribution of each item to the recommendation.
        :param model:
        :param item_id:
        :param user_id:
        :return: returns a dataframe with the contribution to the recommendation of each previously interacted with item.
        """
        #gaya men constructor el explainer (super) w heta num of items fl train dataset
        current_interactions = np.zeros(self.num_items) # [0,0,0,0,.......]
        current_interactions[self.get_user_items(user_id)] = 1 
        # [0,0,0,0,1,1,0,1,1,0......] hanhot 1's makan el hagat ely el user 3amalaha rste already (watched)

        c_u = np.diag(current_interactions) #ba2a 3andy matrix el diagonal beta3ha el content beta3 current interactions

        y_t = self.model.item_embedding().transpose()
        temp = np.matmul(y_t, c_u)
        temp = np.matmul(temp, self.model.item_embedding())
        temp = temp + np.diag([self.model.reg_term] * self.model.latent_dim)
        
        if len(self.get_user_items(user_id)) > 1:
            weight_mtr = np.linalg.inv(temp)
        else:
            weight_mtr = np.linalg.pinv(temp)

        temp = np.matmul(self.model.item_embedding(), weight_mtr)
        
        #I think sim to rec di calculation bethseb el contribution beta3et kol movie fl recommendation choice

        sim_to_rec_id = temp.dot(self.model.item_embedding()[item_id, :])

        sim_to_rec_id = sim_to_rec_id[self.get_user_items(user_id)]

        contribution = {"item": self.get_user_items(user_id), "contribution": sim_to_rec_id}
        contribution = pd.DataFrame(contribution)
        contribution = contribution.sort_values(by=["contribution"], ascending=False)
        return {"item": contribution.item[:self.number_of_contributions],
                "contribution": contribution.contribution[:self.number_of_contributions]}
    
    #this return is the segments of the pie chart el homa beykono column fl table esmo explanations w beyb2a 3obara 3an {item: [27,2763,716,1623,.....], contribution: [0.2,0.345,0.876.....] w betetargem le pie chart ba3d keda fl example
