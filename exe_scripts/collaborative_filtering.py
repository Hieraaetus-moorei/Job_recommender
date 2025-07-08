# Just for debug purpose
import lenskit
lenskit.__version__

import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
import supabase

from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.basic import Bias, Popular
# from lenskit.metrics import topn
from lenskit import batch, topn
from lenskit.topn import RecListAnalysis
from lenskit import crossfold as xf

from typing import List, Dict, Any, Optional, Union

class CollaborativeJobRecommender(Recommender):
    '''
    Collaborative filtering job recommender using LensKit
    '''

    def __init__(self, method = 'user-user', user_k = 20, item_k = 20, min_sim = 0.1, features = 20, reg = 0.1):
        '''
        Initialise the collaborative filtering recommender

        Args:
            method: The CF algorithm to use ('user-user', 'item-item', or 'matrix-factorisation')
            user_k: Number of neighbours for user-based CF
            item_k: Number of neighbours for item-based CF
            min_sim: Minimum similarity threshold
            features: Number of latent features for matrix factorisation
            reg: Regularisation parameter for matrix factorisation
        '''

        self.method = method
        self.user_k = user_k
        self.item_k = item_k
        self.min_sim = min_sim
        self.features = features
        self.reg = reg
        self.id_col = 'job_id'

        # Select algorithm based on method
        if method == 'user-user':
            self.algo = UserUser(nnbrs = user_k, min_sim = min_sim)
        elif method == 'item-item':
            self.algo = ItemItem(nnbrs = item_k, min_sim = min_sim)
        elif method == 'matrix-factorisation':
            self.algo = BiasedMF(features = features, reg = reg)
        else:
            # Default to a popularity baseline if method is not recognised
            self.algo = Popular()

        # Initialise a bias model for better predictions
        self.bias = Bias() # finding bias of users or jobs

    def fit(self, ratings):
        '''
        Train the recommender on the provided ratings dataframe

        Args:
            ratings: DataFrame with columns 'user', 'item', 'rating'
        '''

        # First fit the bias model
        self.bias.fit(ratings)

        # Then fit the main algorithm
        self.algo.fit(ratings)

        # Store ratings data for later use
        self.ratings_df = ratings

        # Keep track of unique users and items
        self.users = ratings['user'].unique()
        self.items = ratings['item'].unique()

        return self

    def recommend(self, user_id, n=10, candidates=None, ratings=None):
        '''
        Recommend jobs to a user based on collaborative filtering.
        This method now directly uses the trained algorithm for prediction.
        '''
        # 1. cold start handling: use cold start strategy if user is not in training data
        if user_id not in self.users:
            # Cold start: The _handle_cold_start method already provides a fallback.
            # print(f'User {user_id} not in training data. Using cold-start strategy.')
            return self._handle_cold_start(user_id, n, candidates)

        # 2. generate canditates
        # exclue items that the user has already interacted with if no candidate
        if candidates is None:
            user_interacted_items = self.ratings_df[self.ratings_df['user'] == user_id]['item'].unique()
            candidates = np.setdiff1d(self.items, user_interacted_items)

        # return empty DataFrame if no candidates for recommendation
        if len(candidates) == 0:
            return pd.DataFrame(columns=['user', 'item', 'score', 'rank'])

        # 3. prediction using self.algo
        try:
            # predict_for_user returns a Series containing score
            preds = self.algo.predict_for_user(user_id, candidates)
            
            # remove NaN predictions and sort by score
            preds = preds.dropna().sort_values(ascending = False).head(n)
            
            if preds.empty:
                # use cold start strategy if no valid predictions
                return self._handle_cold_start(user_id, n, candidates)

            # 4. format the output
            recs_df = preds.reset_index()
            recs_df.columns = ['item', 'score']
            recs_df['user'] = user_id
            recs_df['rank'] = range(1, len(recs_df) + 1)
            
            return recs_df[['user', 'item', 'score', 'rank']]

        except Exception as e:
            print(f'Error during prediction for user {user_id}: {e}')
            # use cold start strategy if prediction fails
            return self._handle_cold_start(user_id, n, candidates)

    def _handle_cold_start(self, user_id, n = 10, candidates = None):
        '''
        Handle cold start problem for new users or when recommendations fail
        '''
        # Use the bias model to provide baseline recommendations

        # if no candidate -> use all items
        if candidates is None:
            candidates = self.items

        # Calculate bias scores for each candidate
        scores = []
        for item in candidates:
            try:
                score = self.bias.score(user_id, item)
                scores.append({'user': user_id, 'item': item, 'score': score})
            except:
                # If bias scoring fails, assign a neutral score
                scores.append({'user': user_id, 'item': item, 'score': 0.0})

        # Convert to DataFrame and rank by score
        recs_df = pd.DataFrame(scores)
        if not recs_df.empty:
            recs_df = recs_df.sort_values('score', ascending = False).head(n)
            # assign ranks based on sorting result
            recs_df['rank'] = range(1, len(recs_df) + 1)
            return recs_df
        else:
            # Return empty DataFrame with correct columns if no scores are available
            return pd.DataFrame(columns = ['user', 'item', 'score', 'rank'])


def prepare_ratings_data(jobs_df, include_applicant_data = True):
    '''
    Prepare ratings data from jobs DataFrame

    Args:
        jobs_df: DataFrame containing job listings with applicant data
        include_applicant_data: Whether to include applicant data fields

    Returns:
        DataFrame with columns 'user', 'item', 'rating'
    '''

    ratings_data = []

    # Process each job to extract applicant data
    for _, job in jobs_df.iterrows():
        job_id = job[job_id_col]

        # Extract applicant data (could be JSON string or already parsed)
        if 'applicants' in job and pd.notna(job['applicants']):
            try:
                applicants = job['applicants']
                if isinstance(applicants, str):
                    try:
                        applicants = json.loads(applicants)
                    except:
                        # Try to split by comma if JSON parsing fails
                        if ',' in applicants:
                            applicants = applicants.split(',')
                        else:
                            applicants = [applicants]

                # Handle different applicant data formats
                if isinstance(applicants, list):
                    for applicant_id in applicants:
                        ratings_data.append({
                            'user': str(applicant_id).strip(),
                            'item': job_id,
                            'rating': 1.0  # An application is a positive signal
                        })
                elif isinstance(applicants, dict):
                    for applicant_id, details in applicants.items():
                        rating = 1.0  # Default positive rating

                        # If details contain rating information, use that
                        if isinstance(details, dict) and 'rating' in details:
                            rating = float(details['rating'])

                        ratings_data.append({
                            'user': str(applicant_id).strip(),
                            'item': job_id,
                            'rating': rating
                        })
            except Exception as e:
                print(f'Error processing applicants for job {job_id}: {e}')

    # Additional implicit signals from applicant interaction data
    if include_applicant_data:
        for _, job in jobs_df.iterrows():
            job_id = job[job_id_col]

            # These fields in the job table might contain user preferences or behaviours
            for field in ['apply_education', 'apply_skills', 'apply_major', 'apply_experience']:
                if field in job and pd.notna(job[field]):
                    try:
                        data = job[field]
                        if isinstance(data, str):
                            try:
                                data = json.loads(data)
                            except:
                                continue

                        if isinstance(data, dict):
                            for user_id, value in data.items():
                                # Create weighted ratings based on field values
                                # For example, higher skills match = higher rating
                                weight = 0.5  # Default weight
                                if isinstance(value, (int, float)):
                                    weight = min(max(float(value) / 10.0, 0.1), 1.0)
                                elif isinstance(value, dict) and 'score' in value:
                                    weight = min(max(float(value['score']) / 10.0, 0.1), 1.0)

                                ratings_data.append({
                                    'user': str(user_id).strip(),
                                    'item': job_id,
                                    'rating': weight
                                })
                    except Exception as e:
                        print(f'Error processing {field} for job {job_id}: {e}')

    # Convert to DataFrame and handle duplicates
    ratings_df = pd.DataFrame(ratings_data)
    if not ratings_df.empty:
        # Aggregate duplicate user-item pairs by taking the maximum rating
        ratings_df = ratings_df.groupby(['user', 'item'])['rating'].max().reset_index()

    return ratings_df


def format_job_info(job):
    '''Format job information for display'''

    job_title = job.get('job_title', job.get('job_name', 'Untitled Job'))
    company = job.get('company_name', 'Unknown Company')
    similarity = job.get('score', 0)

    info = f"- {job_title} at {company} (ID: {job['job_id']})"
    info += f'\n  Recommendation Score: {similarity:.4f}'

    # Add more relevant job info
    if 'job_industry' in job and pd.notna(job['job_industry']):
        info += f"\n  Industry: {job['job_industry']}"

    if 'location' in job and pd.notna(job['location']):
        info += f"\n  Location: {job['location']}"

    if 'legal_benefits' in job and pd.notna(job['legal_benefits']):
        info += f"\n  Benefits: {job['legal_benefits']}"

    return info


def evaluate_recommender(jobs_df, ratings_df, method = 'user-user'):
    '''
    Evaluate the recommender using cross-validation

    Args:
        jobs_df: DataFrame containing job data
        ratings_df: DataFrame with user-item interactions
        method: Collaborative filtering method to evaluate

    Returns:
        DataFrame with evaluation metrics
    '''
    # Define algorithms to evaluate
    algorithms = {
        'Popular': Popular(),
        'BiasModel': Bias(),
        method: CollaborativeJobRecommender(method = method)
    }

    # Set up evaluation
    eval_results = []

    # Create 5-fold cross-validation splits
    for train, test in xf.partition_users( ratings_df, 5, xf.SampleFrac(0.2) ):
        # Clone the jobs_df to avoid modifying the original
        train_jobs = jobs_df.copy()

        # Create 'candidates' set - all items in the test set
        candidates = test['item'].unique()

        # Create a function to filter candidates
        def candidates_func(user):
            return candidates

        # For each algorithm
        for name, algo in algorithms.items():
            # Train the algorithm
            algo.fit(train)

            # Generate recommendations for test users
            # Use the candidates_func for filtering
            recs = batch.recommend(algo, test['user'].unique(), 10, candidates = candidates_func)

            # Recommendation List Analysis (RLA)
            rla = topn.RecListAnalysis()
            rla.add_metric(topn.ndcg)  # Example: Add ndcg metric
            # You can add other metrics like precision, recall, etc.
            _metrics = rla.compute(recs, test)

            # Compute evaluation metrics
            # _metrics = topn.compute_metrics(test, recs, include_missing = True)

            # choose certain algorithm
            _metrics['Algorithm'] = name
            eval_results.append(_metrics)

    # Combine and return all results
    return pd.concat(eval_results)


def main():
    '''
    Main function to run the collaborative filtering recommender
    '''

    print('Initialising collaborative job recommender system...')

    # Load environment variables
    load_dotenv()

    # Retrieve the url and API key from environment
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')

    # Check whether the setup goes smoothly
    if supabase_key:
        print('Supabase key loaded successfully.')
    else:
        print("Supabase key not found. Please make sure it's defined as a secret.")
        return

    # Connect to supabase
    supabase_client = supabase.create_client(supabase_url, supabase_key)

    # Get all jobs from supabase
    jobs_data = supabase_client.from_('jobs').select('*').execute().data

    # Load the data into a DataFrame
    jobs_df = pd.DataFrame(jobs_data)

    # in case there is nothing
    if jobs_df.empty:
        print('Error: No job data was loaded from Supabase.')
        return

    print(f'Loaded {len(jobs_df)} jobs from database.')

    # Define global ID column name
    global job_id_col
    job_id_col = 'job_id'

    # Prepare ratings data from job applications
    print('Preparing ratings data from job applications...')
    ratings_df = prepare_ratings_data(jobs_df)

    # in case rating is missed
    if ratings_df.empty:
        print('Error: No ratings data could be extracted.')
        return

    print(f'Created {len(ratings_df)} user-job interactions.')

    # Initialise and train the recommender
    print('Training collaborative filtering recommenders...')

    # Create three different recommenders for comparison
    user_user_rec = CollaborativeJobRecommender(method = 'user-user')
    item_item_rec = CollaborativeJobRecommender(method = 'item-item')
    mf_rec = CollaborativeJobRecommender(method = 'matrix-factorisation')

    # Train all recommenders
    user_user_rec.fit(ratings_df)
    item_item_rec.fit(ratings_df)
    mf_rec.fit(ratings_df)

    print('Recommender systems trained successfully.')

    # Example 1: Find users with most applications for testing
    user_counts = ratings_df['user'].value_counts()
    test_user_id = user_counts.index[0]  # User with most applications
    test_count = user_counts.iloc[0]
    print(f'\nTest user {test_user_id} has applied to {test_count} jobs')

    # Get recommendations from each method
    for name, rec in [('User-User CF', user_user_rec),
                      ('Item-Item CF', item_item_rec),
                      ('Matrix Factorisation', mf_rec)]:
        print(f'\nGenerating recommendations using {name}...')
        # Get recommendations for the test user (top n)
        recommendations = rec.recommend(test_user_id, n = 3)

        if recommendations.empty:
            print(f'No recommendations found for user {test_user_id} using {name}.')
        else:
            print(f'Top 3 job recommendations for user {test_user_id} using {name}:')
            for _, rec_row in recommendations.iterrows():
                job_id = rec_row['item']
                job = jobs_df[jobs_df[job_id_col] == job_id]

                if not job.empty:
                    job_dict = job.iloc[0].to_dict()
                    job_dict['score'] = rec_row['score']
                    print(format_job_info(job_dict) + '\n')

    # Example 2: Create a test case for a new user with one job application
    print('\nSimulating a new user applying to a job...')
    new_user_id = 'new_test_user_789'

    # Pick a random job for the test
    test_job_id = jobs_df.iloc[5][job_id_col]  # A different job than first example

    print(f'New user {new_user_id} applied to job {test_job_id}')

    # Create a test ratings DataFrame for this new user
    test_ratings = pd.DataFrame({
        'user': [new_user_id],
        'item': [test_job_id],
        'rating': [1.0]
    })

    # Combine with existing ratings
    combined_ratings = pd.concat([ratings_df, test_ratings])

    # Train a new recommender with the combined data
    test_rec = CollaborativeJobRecommender(method = 'item-item')
    test_rec.fit(combined_ratings)

    # Get recommendations for the new user
    recommendations = test_rec.recommend(new_user_id, n = 3)

    if recommendations.empty:
        print('No recommendations found for the new user.')
    else:
        print(f'Top 3 job recommendations for the new user:')
        for _, rec_row in recommendations.iterrows():
            job_id = rec_row['item']
            job = jobs_df[jobs_df[job_id_col] == job_id]

            if not job.empty:
                job_dict = job.iloc[0].to_dict()
                job_dict['score'] = rec_row['score']
                print(format_job_info(job_dict) + '\n')

    # Optionally run evaluation
    print('\nWould you like to run evaluation? This may take some time. (y/n)')
    response = input()
    if response.lower() == 'y':
        print('Running recommender evaluation...')
        eval_results = evaluate_recommender(jobs_df, ratings_df, method = 'user-user')
        print('\nEvaluation Results:')
        print(eval_results.groupby('Algorithm').mean())

if __name__ == '__main__':
    main()