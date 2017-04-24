import pandas as pd
import numpy as np

def item_base_collaborative_filter():
    r_cols = ['user_id', 'movie_id', 'rating']
    ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=r_cols, usecols=range(3), encoding="ISO-8859-1")
    m_cols = ['movie_id', 'title']
    movies = pd.read_csv("ml-100k/u.item", sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")
    ratings = pd.merge(movies, ratings)

    #Create a new movie table with the rows = user_id, columns = title, and the values are the rating
    movie_ratings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')

    #Getting simillar movies to starwars, not so good because include videos with less than 100 rates (Obscure Movies)
    star_war_ratings = movie_ratings['Star Wars (1977)']
    similar_movies = movie_ratings.corrwith(star_war_ratings)
    similar_movies = similar_movies.dropna()
    #similar_movies.sort_values(inplace=True, ascending= False)

    #Grouping by mean and size and removing the movies with less than 100 rates.
    movie_stats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
    popular_movies = movie_stats['rating']['size'] >= 100
    #movie_stats = movie_stats[popular_movies].sort_values([('rating', 'mean')], ascending=False) #Shows the movies with the best rates on top.

    df = movie_stats[popular_movies].join(pd.DataFrame(similar_movies, columns=['similarity']))

    #filtered_value = df.sort_values(['similarity'], ascending=False)
    print(df.sort_values(['similarity'], ascending=False)[:15])


item_base_collaborative_filter()
