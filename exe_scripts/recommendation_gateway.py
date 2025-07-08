import pandas as pd
import os
import supabase
from dotenv import load_dotenv

# import predefined recommenders
from content_based_filtering import ContentBasedJobRecommender
from collaborative_filtering import CollaborativeJobRecommender, prepare_ratings_data

class RecommendationGateway:
    '''
    gateway for hybrid recommendation system, dynamically adjusts weights between content-based and collaborative filtering
    '''

    def __init__(self, cold_start_threshold = 5, hot_start_threshold = 20):
        '''
        initialise the gateway

        Args:
            cold_start_threshold (int): define the upper limit of user interactions for cold start state
            hot_start_threshold (int): define the lower limit of user interactions for hot start state
        '''
        print('Initialising Recommendation Gateway...')
        self.cb_recommender = ContentBasedJobRecommender()
        # choose a collaborative filtering algorithm as default e.g. 'item-item'
        self.cf_recommender = CollaborativeJobRecommender(method = 'item-item')

        self.cold_start_threshold = cold_start_threshold
        self.hot_start_threshold = hot_start_threshold

        self.jobs_df = None
        self.ratings_df = None
        
        # default weights
        self.cold_weights = {'cb': 0.9, 'cf': 0.1}
        self.hot_weights = {'cb': 0.1, 'cf': 0.9}

    def fit(self, jobs_df, ratings_df):
        '''
        train the underlying recommendation models

        Args:
            jobs_df (pd.DataFrame): the dataframe containing all job information
            ratings_df (pd.DataFrame): the dataframe containing user interaction records ('user', 'item', 'rating'）
        '''
        print('Fitting models...')
        self.jobs_df = jobs_df
        self.ratings_df = ratings_df

        # train content-based model
        print('Training Content-Based Recommender...')
        self.cb_recommender.fit(self.jobs_df)

        # train collaborative filtering model
        print('Training Collaborative Filtering Recommender...')
        self.cf_recommender.fit(self.ratings_df)
        print('All models fitted successfully')

    def _get_user_state_weights(self, user_id):
        '''
        dynamically calculate the weights for CB and CF based on user interaction count
        '''
        if self.ratings_df is None:
            return self.cold_weights

        user_ratings_count = self.ratings_df[self.ratings_df['user'] == user_id].shape[0]

        # cold start state
        if user_ratings_count < self.cold_start_threshold:
            print(f"User '{user_id}' is in COLD START state ({user_ratings_count} interactions).")
            return self.cold_weights

        # hot start state
        if user_ratings_count >= self.hot_start_threshold:
            print(f"User '{user_id}' is in HOT START state ({user_ratings_count} interactions).")
            return self.hot_weights

        # warm start (dynamic adjustment)
        print(f"User '{user_id}' is in WARM START state ({user_ratings_count} interactions).")
        # calculate the progress in warm start (0.0 to 1.0)
        progress = (user_ratings_count - self.cold_start_threshold) / (self.hot_start_threshold - self.cold_start_threshold)
        
        # calculate CF weight using linear interpolation
        cf_weight = self.cold_weights['cf'] + progress * (self.hot_weights['cf'] - self.cold_weights['cf'])
        cb_weight = 1.0 - cf_weight
        
        return {'cb': cb_weight, 'cf': cf_weight}

    def recommend(self, user_id, n = 10):
        '''
        generate hybrid recommendation list for the specified user

        Args:
            user_id (str): user ID
            n (int): number of recommendations to generate

        Returns:
            pd.DataFrame: DataFrame containing the final recommendations
        '''
        # 1. get user state weights
        weights = self._get_user_state_weights(user_id)
        print(f"Calculated weights: Content-Based: {weights['cb']:.2f}, Collaborative: {weights['cf']:.2f}")

        # 2. get recommendations from collaborative filtering
        # get more recommendations than needed for merging later
        cf_recs = self.cf_recommender.recommend(user_id, n = n * 2) 

        # 3. get recommendations from content-based filtering
        # CB recommendations are based on items the user has interacted with
        user_applied_jobs = self.ratings_df[self.ratings_df['user'] == user_id]['item'].unique()
        
        all_cb_recs = []
        if len(user_applied_jobs) > 0:
            for job_id in user_applied_jobs:
                # find similar jobs for each job the user has applied to
                similar_jobs = self.cb_recommender.get_similar_jobs(job_id, n = n)
                all_cb_recs.extend(similar_jobs)

            # turn into DataFrame format
            cb_recs_df = pd.DataFrame(all_cb_recs)
            # sort by similarity score and remove duplicates
            if not cb_recs_df.empty:
                cb_recs_df = cb_recs_df.sort_values('similarity_score', ascending = False).drop_duplicates(subset = 'job_id')
                cb_recs_df = cb_recs_df.rename(columns={'job_id': 'item', 'similarity_score': 'score'})
                # remove jobs the user has already applied to
                cb_recs_df = cb_recs_df[~cb_recs_df['item'].isin(user_applied_jobs)]
            else:
                cb_recs_df = pd.DataFrame(columns = ['item', 'score'])
        else:
            cb_recs_df = pd.DataFrame(columns = ['item', 'score'])


        # 4. combine and weight
        # normalise scores (Min-Max Scaling to 0-1 range)
        if not cf_recs.empty and cf_recs['score'].max() > 0:
            cf_recs['normalized_score'] = cf_recs['score'] / cf_recs['score'].max()
        else:
            cf_recs['normalized_score'] = 0

        if not cb_recs_df.empty and cb_recs_df['score'].max() > 0:
            cb_recs_df['normalized_score'] = cb_recs_df['score'] / cb_recs_df['score'].max()
        else:
            cb_recs_df['normalized_score'] = 0
            
        # combine results for weighted score calculation using dictionary
        combined_scores = {}
        
        for _, row in cf_recs.iterrows():
            combined_scores[row['item']] = row['normalized_score'] * weights['cf']
            
        for _, row in cb_recs_df.iterrows():
            if row['item'] in combined_scores:
                combined_scores[row['item']] += row['normalized_score'] * weights['cb']
            else:
                combined_scores[row['item']] = row['normalized_score'] * weights['cb']

        if not combined_scores:
            print('No recommendations could be generated')
            return pd.DataFrame()

        # 5. sort and return results
        final_recs_df = pd.DataFrame(list( combined_scores.items() ), columns = ['item', 'final_score'])
        final_recs_df = final_recs_df.sort_values('final_score', ascending = False).head(n)
        final_recs_df['rank'] = range(1, len(final_recs_df) + 1)
        
        # append detailed info for display
        final_recs_df = final_recs_df.merge(self.jobs_df, left_on = 'item', right_on = 'job_id', how = 'left')

        return final_recs_df

def main():
    '''
    main function to demonstrate the recommendation gateway
    '''
    # --- load data ---
    load_dotenv()
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')

    if not supabase_key:
        print('Supabase key not found')
        return

    supabase_client = supabase.create_client(supabase_url, supabase_key)
    jobs_data = supabase_client.from_('jobs').select('*').execute().data
    jobs_df = pd.DataFrame(jobs_data)
    
    if jobs_df.empty:
        print('Error: No job data loaded.')
        return

    print(f'Loaded {len(jobs_df)} jobs from database')
    
    # in case collaborative_filtering.py's job_id_col is not set
    import collaborative_filtering
    collaborative_filtering.job_id_col = 'job_id'
    
    ratings_df = prepare_ratings_data(jobs_df)
    
    if ratings_df.empty:
        print('Error: No ratings data could be extracted')
        return
        
    print(f'Created {len(ratings_df)} user-job interactions')

    # --- initialise and train the gateway system ---
    gateway = RecommendationGateway(cold_start_threshold = 5, hot_start_threshold = 20)
    gateway.fit(jobs_df, ratings_df)
    
    # --- test cases ---
    user_counts = ratings_df['user'].value_counts()
    
    # case 1: hot start user (most interactions)
    hot_user_id = user_counts.index[0]
    print(f'\n--- TestCase 1: HOT START User ---')
    recommendations = gateway.recommend(hot_user_id, n = 5)
    print(f"\nTop 5 Hybrid Recommendations for Hot User '{hot_user_id}':")
    if not recommendations.empty:
        for _, rec in recommendations.iterrows():
            print(f"- {rec['job_title']} at {rec['company_name']} (Final Score: {rec['final_score']:.4f})")
    
    # case 2: warm start user (moderate interactions)
    warm_user_id = user_counts[(user_counts > 5) & (user_counts < 20)].index[0] if any( (user_counts > 5) & (user_counts < 20) ) else None
    if warm_user_id:
        print(f'\n--- TestCase 2: WARM START User ---')
        recommendations = gateway.recommend(warm_user_id, n = 5)
        print(f"\nTop 5 Hybrid Recommendations for Warm User '{warm_user_id}':")
        if not recommendations.empty:
            for _, rec in recommendations.iterrows():
                print(f"- {rec['job_title']} at {rec['company_name']} (Final Score: {rec['final_score']:.4f})")
    
    # case 3: cold start user (least interactions)
    cold_user_id = user_counts.index[-1]
    print(f'\n--- TestCase 3: COLD START User ---')
    recommendations = gateway.recommend(cold_user_id, n = 5)
    print(f"\nTop 5 Hybrid Recommendations for Cold User '{cold_user_id}':")
    if not recommendations.empty:
        for _, rec in recommendations.iterrows():
            print(f"- {rec['job_title']} at {rec['company_name']} (Final Score: {rec['final_score']:.4f})")


if __name__ == '__main__':
    main()