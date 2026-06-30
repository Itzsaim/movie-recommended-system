import pandas as pd
import pickle

# CSV file load karo
movies = pd.read_csv('tmdb_5000_movies.csv')

# Dictionary create karo
movies_dict = movies.to_dict('records')

# Save as pickle
with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(movies_dict, f)

print("movie_dict.pkl generated successfully!")
