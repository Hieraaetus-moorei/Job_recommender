# Just for debug purpose
import lenskit
lenskit.__version__

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Bias
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from typing import List, Dict, Any
import supabase
import os
from dotenv import load_dotenv

# load the file you set up earlier and dump it into environment
load_dotenv()

# retrieve the url and API key from environment and assign them to variables
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')

# check whether the setup goes smoothly
if supabase_key:
    print('Supabase key loaded successfully.')
else:
    print("Supabase key not found. Please make sure it's defined as a secret.")

# connect to supabase
supabase_client = supabase.create_client(supabase_url, supabase_key)
# get all jobs from supabase
jobs_data = supabase_client.from_('jobs').select('*').execute().data

# Load the data into a DataFrame
jobs_df = pd.DataFrame(jobs_data)

# setting up customised recommender
class ContentBasedJobRecommender(Recommender):
    '''
    Content-based job recommender using TF-IDF and cosine similarity
    '''

    def __init__(self):
        self.jobs_df = None
        self.tfidf_matrix = None
        self.feature_columns = [
            'job_industry',
            'industry',
            'primary_category',
            'job_title',
            'job_name',
            'job_description',
            'job_category',
            'location',
            'skills',
            'tools'
        ]

        # Fields for applicant data analysis
        self.applicant_fields = [
            'apply_education',
            'apply_gender',
            'apply_language',
            'apply_age_distribution',
            'apply_experience',
            'apply_major',
            'apply_skills',
            'apply_certificates'
        ]
        self.id_col = 'job_id'
        self.bias = Bias()

    def fit(self, jobs_df):
        '''
        Train the recommender on the provided jobs dataframe
        '''
        self.jobs_df = jobs_df.copy()

        # Create a combined text field for TF-IDF
        self.jobs_df['combined_features'] = self.jobs_df.apply(self._combine_features, axis = 1)

        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words = 'english')
        self.tfidf_matrix = tfidf.fit_transform(self.jobs_df['combined_features'].fillna(''))

        # Store cosine similarity matrix for faster recommendations
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Create a dummy ratings DataFrame for Bias fitting
        ratings_data = {
            'user': ['dummy_user'] * len(self.jobs_df),
            'item': self.jobs_df[self.id_col].tolist(),
            'rating': [1.0] * len(self.jobs_df)
        }
        ratings_df = pd.DataFrame(ratings_data)
        self.bias.fit(ratings_df)

        return self

    def _combine_features(self, row):
        '''
        Combine relevant features into a single string for text processing
        '''
        combined = ''

        # Process regular text fields
        for col in self.feature_columns:
            if col in row.index and pd.notna(row[col]):
                if isinstance(row[col], str):
                    combined += row[col] + ' '

        # Process benefits fields (they may be empty)
        if 'legal_benefits' in row.index and pd.notna(row['legal_benefits']):
            combined += row['legal_benefits'] + ' '

        if 'other_benefits' in row.index and pd.notna(row['other_benefits']):
            combined += row['other_benefits'] + ' '

        # Process applicant data fields to understand job requirements
        for field in self.applicant_fields:
            if field in row.index and pd.notna(row[field]):
                try:
                    # Try to parse JSON data
                    if isinstance(row[field], str):
                        data = json.loads(row[field])
                    else:
                        data = row[field]

                    if isinstance(data, dict):
                        # Extract all keys and values from the dictionary
                        for key, value in data.items():
                            combined += f'{key} {value} '
                except:
                    # If parsing fails, just use as is
                    if isinstance(row[field], str):
                        combined += row[field] + ' '

        return combined

    def get_similar_jobs(self, job_id, n = 10):
        '''
        Get n most similar jobs to the given job_id
        '''
        if self.jobs_df is None or self.tfidf_matrix is None:
            raise ValueError('Recommender has not been trained yet')

        # Find the index of the job in our dataframe
        idx = self.jobs_df.index[self.jobs_df[self.id_col] == job_id].tolist()
        if not idx:
            return []
        idx = idx[0]

        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse = True)

        # Remove the job itself
        sim_scores = [s for s in sim_scores if s[0] != idx]

        # Get top N similar jobs
        sim_scores = sim_scores[:n]
        job_indices = [i[0] for i in sim_scores]

        # Return the similar jobs with their similarity scores
        result = []
        for i, score in enumerate(sim_scores):
            job_idx = job_indices[i]
            job_data = self.jobs_df.iloc[job_idx].to_dict()
            job_data['similarity_score'] = score[1]
            result.append(job_data)

        return result

    # candidates: recommend jobs from given options
    # ratings: applicant might rate some jobs in the past
    def recommend(self, user_id, n = 10, candidates = None, ratings = None):
        '''
        Recommend jobs to a user based on their application history (implements the LensKit Recommender interface)
        '''
        if self.jobs_df is None or self.tfidf_matrix is None:
            raise ValueError('Recommender has not been trained yet')

        # Find jobs this user has applied to (to avoid repetitive recommendation)
        if ratings is not None and not ratings.empty:
            applied_jobs = ratings[ratings['user'] == user_id]['item'].unique()
        else:
            # Try to extract from the applicants field in jobs_df
            applied_jobs = []
            for _, job in self.jobs_df.iterrows():
                if pd.notna(job.get('applicants')):
                    try:
                        applicants = job['applicants']
                        if isinstance(applicants, str):
                            try:
                                applicants = json.loads(applicants)
                            except:
                                # If JSON parsing fails, try to check if user_id is in the string
                                if str(user_id) in applicants:
                                    applied_jobs.append(job[self.id_col])
                                continue

                        # Check if user_id is in applicants list or dict
                        if isinstance(applicants, list) and str(user_id) in [str(a) for a in applicants]:
                            applied_jobs.append(job[self.id_col])
                        elif isinstance(applicants, dict) and str(user_id) in applicants:
                            applied_jobs.append(job[self.id_col])
                    except Exception as e:
                        print(f'Error processing applicants for job {job[self.id_col]}: {e}')

        if not applied_jobs:
            return pd.DataFrame(columns = ['user', 'item', 'score', 'rank'])

        # Get all similar jobs for the applied jobs
        all_recommendations = []
        for job_id in applied_jobs:
            try:
                similar_jobs = self.get_similar_jobs(job_id, n = 20)  # Get more than needed
                all_recommendations.extend(similar_jobs)
            except Exception as e:
                print(f'Error getting similar jobs for {job_id}: {e}')

        # Remove duplicates and already applied jobs
        unique_recs = {}
        for rec in all_recommendations:
            job_id = rec[self.id_col]
            if job_id not in applied_jobs and (job_id not in unique_recs or rec['similarity_score'] > unique_recs[job_id]['similarity_score']):
                unique_recs[job_id] = rec

        # Sort by similarity score and take top N
        sorted_recs = sorted(unique_recs.values(), key = lambda x: x['similarity_score'], reverse = True)[:n]

        # Format for LensKit
        results = []
        for i, rec in enumerate(sorted_recs):
            results.append({
                'user': user_id,
                'item': rec[self.id_col],
                'score': rec['similarity_score'],
                'rank': i + 1
            })

        return pd.DataFrame(results)


def format_job_info(job):
    '''Format job information for display'''
    job_title = job.get('job_title', job.get('job_name', 'Untitled Job'))
    company = job.get('company_name', 'Unknown Company')
    similarity = job.get('similarity_score', 0)

    info = f"- {job_title} at {company} (ID: {job['job_id']})"
    info += f'\n  Similarity Score: {similarity:.4f}'

    # Add more relevant job info
    if 'job_industry' in job and pd.notna(job['job_industry']):
        info += f"\n  Industry: {job['job_industry']}"

    if 'location' in job and pd.notna(job['location']):
        info += f"\n  Location: {job['location']}"

    if 'legal_benefits' in job and pd.notna(job['legal_benefits']):
        info += f"\n  Benefits: {job['legal_benefits']}"

    return info


def main():
    print('Initialising job recommender system...')

    # Check if DataFrame is loaded successfully
    if jobs_df.empty:
        print('Error: No job data was loaded from Supabase.')
        return

    print(f'Loaded {len(jobs_df)} jobs from database.')

    # Initialise and train the recommender
    recommender = ContentBasedJobRecommender()
    recommender.fit(jobs_df)
    print('Recommender system trained successfully.')



    # Example 1: Get similar jobs to a specific job
    sample_job_id = jobs_df.iloc[0]['job_id']  # Get the first job ID as a sample
    print(f'\nFinding similar jobs to job ID: {sample_job_id}')

    similar_jobs = recommender.get_similar_jobs(sample_job_id, n = 3)
    print(f'Top 3 similar jobs to {sample_job_id}:')
    for job in similar_jobs:
        print(format_job_info(job) + '\n')

    # Example 2: Create a test user ID and add them as an applicant to a job
    print('\nSimulating a user applying to a job...')
    test_user_id = 'test_user_123'

    # This is just for testing - in a real app, you'd use actual applicant data
    print('Generating recommendations for a test user who applied to job:', sample_job_id)

    # Create a test ratings DataFrame to simulate the user having applied to this job
    test_ratings = pd.DataFrame({
        'user': [test_user_id],
        'item': [sample_job_id],
        'rating': [1.0]
    })

    # Get recommendations for this test user
    recommendations = recommender.recommend(test_user_id, n = 3, ratings = test_ratings)

    if recommendations.empty:
        print('No recommendations found for test user.')
    else:
        print(f'Top 3 job recommendations for test user:')
        for _, rec in recommendations.iterrows():
            job_id = rec['item']
            job = jobs_df[jobs_df['job_id'] == job_id].iloc[0].to_dict()
            job['similarity_score'] = rec['score']
            print(format_job_info(job) + '\n')

if __name__ == '__main__':
    main()