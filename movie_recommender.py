import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

#print(movies.head())
#print(credits.head())
#merging two datasets movies and credits

movies = movies.merge(credits, on='title')
print(movies.head())
movies.info()
#Only important columns

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
#print(movies.head())
#print("finding null values")

#print(movies.isnull().sum()) #finding null values
#print("dropping null values")
movies.dropna(inplace=True) #dropping null values
#print(movies.isnull().sum())
#print(movies.duplicated().sum())#finding duplicate values

# Genres aur Keywords ko list of names me convert karega
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):# string ko Python object (list/dict) me convert
        l.append(i['name'])# har dict ka 'name' nikal kar list me daal do
    return l
# Cast se sirf top 3 actors nikalne ke liye
def convert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l
# Crew se sirf Director nikalne ke liye
def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':# agar crew member ka job Director hai
            l.append(i['name'])
            break
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
# Spaces hata do sabhi names se (taake Science Fiction -> ScienceFiction ban jaye)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

print(movies.head())

#create new colums in mein concat kar dege(overview,genres,cast,keywords,crew)
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
print(movies.head())
#ab humein yah three colums chahiya only
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)) # convert string
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower()) #convert lowercase
print(new_df.head())
# convert the text to vertors
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

ps = PorterStemmer()  
# Yahaan hum nltk library se PorterStemmer ka object bana rahe hain.
# PorterStemmer ek algorithm hai jo words ko unke root (stem) form mein convert karta hai.
# Example: "running" -> "run", "loved" -> "love"

def stem(text):  
    # Ek function banaya jiska naam 'stem' hai aur jo ek string (text) lega.

    y = []  
    # Ek empty list banayi jisme hum stemmed words ko store karenge.

    for i in text.split():  
        # Text ko words mein split kar liya (space ke hisaab se). Har word ko 'i' variable mein le kar loop chalaya.

        y.append(ps.stem(i))  
        # Har word pe PorterStemmer lagaya (ps.stem(i)) aur uska stemmed form list 'y' mein add kar diya.

    return " ".join(y)  
    # List ke saare words ko dobara space ke saath join karke ek single string bana kar return kar diya.

new_df['tags'] = new_df['tags'].apply(stem)  
# 'new_df' dataframe ke 'tags' column pe stem function apply kiya.Matlab har row ke 'tags' ke words ko stemmed form mein convert kar diya.

print(new_df['tags'])  
# Stemmed 'tags' column ko print kar diya jisse hum result dekh saken.

similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
new_df['title'].values
pickle.dump(similarity,open('similarity.pkl','wb'))



