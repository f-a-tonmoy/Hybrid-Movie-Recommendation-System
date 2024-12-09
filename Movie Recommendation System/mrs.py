import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


user_item_matrix, movie_db, ratings = None, None, None


def load_data():
    global user_item_matrix, movie_db, ratings

    user_item_matrix = pd.read_csv('user_item_matrix.csv')
    user_item_matrix.set_index('userId', inplace=True)
    user_item_matrix.columns = user_item_matrix.columns.astype(int)

    ratings = pd.read_csv('filtered_ratings.csv')
    movie_db = pd.read_csv('movie_db.csv')

    matrix_info()
    return user_item_matrix, movie_db, ratings


def matrix_info():
    print('User-item matrix', user_item_matrix.shape)
    print('Ratings', ratings.shape)
    print('Movies', movie_db.shape)

    rows, cols = user_item_matrix.shape
    total_cells = rows * cols
    missing_cells = user_item_matrix.isnull().sum().sum()

    print(f'Users: {rows}')
    print(f'Movies: {cols}')
    print(f'Total cells: {total_cells}')
    print(f'Missing cells: {missing_cells}')
    print(f'Missing percenatge: {missing_cells/total_cells:.2%}')


def check_user_id(user_id):
    return user_id in user_item_matrix.index


def pearson_corr(user_id):
    user_item = user_item_matrix.copy()
    row_means = user_item.mean(axis=1, skipna=True)
    user_item = user_item.subtract(row_means, axis=0)

    target_user_ratings = user_item.loc[user_id]
    corr_data = []

    for index, other_user_ratings in user_item.iterrows():
        if user_id == index:
            continue

        common_ratings = target_user_ratings.notna() & other_user_ratings.notna()

        numerator = (target_user_ratings[common_ratings]
                     * other_user_ratings[common_ratings]).sum()
        denominator = np.sqrt((target_user_ratings[common_ratings] ** 2).sum()) * \
            np.sqrt((other_user_ratings[common_ratings] ** 2).sum())

        if np.isclose(denominator, 0):
            continue

        corr_data.append({
            'user1Id': user_id,
            'user2Id': index,
            'correlation': numerator / denominator
        })

    return pd.DataFrame(corr_data)


def recommend_movies(correlations, n):
    user_id = correlations.iloc[0, 0]
    top_neighbors = correlations.nlargest(n * 5, 'correlation')
    neighbor_ratings = pd.merge(
        top_neighbors, ratings, left_on='user2Id', right_on='userId')
    avg_movie_ratings = neighbor_ratings.groupby(
        'movieId')['rating'].mean().round(2)

    target_user_ratings = ratings[ratings['userId'] == user_id]
    movies_watched = target_user_ratings['movieId'].unique()
    recommendable_movies = avg_movie_ratings[~avg_movie_ratings.index.isin(
        movies_watched)].to_frame()

    top_neighbors_ids = top_neighbors['user2Id']
    correlation_values = top_neighbors['correlation']

    pred_ratings = []

    for idx in recommendable_movies.index:
        mov_ratings = user_item_matrix[idx].loc[top_neighbors_ids]

        mult_sum, corr_sum = 0, 0
        for rating, corr_val in zip(mov_ratings, correlation_values):
            if np.isnan(rating):
                continue

            mult_sum += (rating * corr_val)
            corr_sum += corr_val

        if corr_sum > 0:
            predicted_rating = mult_sum / corr_sum
        else:
            # print('in')
            predicted_rating = avg_movie_ratings.loc[idx]

        pred_ratings.append(round(predicted_rating, 2))

    recommendable_movies['pred_rating'] = pred_ratings

    top_recommendations = pd.merge(recommendable_movies.nlargest(
        n, 'pred_rating'), movie_db, on='movieId')
    top_recommendations.drop(['overview'], axis=1, inplace=True)

    # print(top_recommendations.columns)
    pred_ratings = top_recommendations.loc[:, ['movieId', 'pred_rating']]
    pred_ratings['actual_rating'] = np.nan

    # print(pred_ratings)
    # print(top_recommendations)

    order = ['movieId', 'title', 'genres', 'rating']
    return top_recommendations.reindex(columns=order), pred_ratings


def jaccard_sim(title):
    target_movie = set(
        str(movie_db[movie_db['title'] == title]['overview']).split(', '))

    result = []
    for _, movie in movie_db.iterrows():
        if movie['title'] == title:
            continue

        other_movie = set(str(movie['overview']).split(', '))
        jaccard_similarity = len(target_movie.intersection(
            other_movie)) / len(target_movie.union(other_movie))
        result.append({
            'target_movie': title,
            'other_movie': movie['title'],
            'jaccard_sim_score': jaccard_similarity
        })

    return pd.DataFrame(result)


def recommend_similar_movies(user_id, similarities, n):
    movies_not_watched = user_item_matrix.columns[user_item_matrix.loc[user_id].isna(
    )]
    movies_not_watched_titles = movie_db[movie_db['movieId'].isin(
        movies_not_watched)]['title'].to_frame()

    recommended_movies = pd.merge(
        similarities, movies_not_watched_titles, left_on='other_movie', right_on='title')
    recommended_movies = pd.merge(recommended_movies, movie_db)

    avg_movie_ratings = ratings.groupby('movieId')['rating'].mean()
    recommended_movies = pd.merge(
        recommended_movies, avg_movie_ratings, on='movieId')

    recommended_movies = recommended_movies.sort_values(
        ['jaccard_sim_score', 'rating'], ascending=False)

    recommended_movies['jaccard_sim_score'] = round(
        recommended_movies['jaccard_sim_score'], 2)
    recommended_movies['rating'] = round(recommended_movies['rating'], 2)

    recommended_movies = recommended_movies.reindex(
        columns=['movieId', 'title', 'genres', 'rating'])

    return recommended_movies.head(n)


def update_rating(user_id, movie_title, rating, user_ratings):
    global user_item_matrix, ratings

    movie_id = movie_db[movie_db['title'] == movie_title]['movieId'].values[0]
    # print(movie_id, type(movie_id))

    user_item_matrix.loc[user_id, movie_id] = rating
    # user_ratings.loc[int(movie_id), 'actual_rating'] = rating
    user_ratings.loc[user_ratings['movieId']
                     == movie_id, 'actual_rating'] = rating

    ratings.loc[len(ratings)] = [user_id, movie_id, rating]
    ratings['userId'] = ratings['userId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)

    user_item_matrix.to_csv('user_item_matrix.csv')
    ratings.to_csv('filtered_ratings.csv', index=False)

    return user_ratings


def eval_performance(rated_movs):
    preds = rated_movs['pred_rating'].values
    actuals = rated_movs['actual_rating'].values

    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    # print(rated_movs)

    rated_movs.loc[:, 'predicted_label'] = rated_movs['pred_rating'] >= 4
    rated_movs.loc[:, 'actual_label'] = rated_movs['actual_rating'] >= 4

    TP = ((rated_movs['predicted_label'] == True) &
          (rated_movs['actual_label'] == True)).sum()
    TN = ((rated_movs['predicted_label'] == False) &
          (rated_movs['actual_label'] == False)).sum()
    FP = ((rated_movs['predicted_label'] == True) & (
        rated_movs['actual_label'] == False)).sum()
    FN = ((rated_movs['predicted_label'] == False) &
          (rated_movs['actual_label'] == True)).sum()

    print(TP, TN, FP, FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f'\nRMSE: {rmse:.2f}')
    print(f'Precision: {precision:.2%}')
    print(f'Recall: {recall:.2%}')
    print(f'Accuracy: {accuracy:.2%}')

    df = pd.read_csv('eval_metrics.csv')
    n, old_rmse = df.loc[0, 'n'], df.loc[0, 'rmse']
    old_precision, old_recall, old_accuracy = df.loc[0,
                                                     'precision'], df.loc[0, 'recall'], df.loc[0, 'accuracy']

    avg_rmse = (n * old_rmse + rmse) / (n + 1)
    avg_precision = (n * old_precision + precision) / (n + 1)
    avg_recall = (n * old_recall + recall) / (n + 1)
    avg_accuracy = (n * old_accuracy + accuracy) / (n + 1)

    print(f'\nAverage RMSE: {avg_rmse:.2f}')
    print(f'Average Precision: {avg_precision:.2%}')
    print(f'Average Recall: {avg_recall:.2%}')
    print(f'Average Accuracy: {avg_accuracy:.2%}')

    data = {
        'n': [n+1],
        'rmse': [avg_rmse],
        'precision': [avg_precision],
        'recall': [avg_recall],
        'accuracy': [avg_accuracy],
    }

    df = pd.DataFrame(data)
    df.to_csv('eval_metrics.csv', index=False)

    values = [rmse, precision, recall, accuracy]
    avg_values = [avg_rmse, avg_precision, avg_recall, avg_accuracy]

    plot_values(values, avg_values)


def plot_values(values, avg_values):
    metrics = ['RMSE', 'Precision', 'Recall', 'Accuracy']
    average_metrics = ['Average RMSE', 'Average Precision',
                       'Average Recall', 'Average Accuracy']

    plt.figure(figsize=(8, 6))

    plt.subplot(3, 1, 1)
    plt.barh([metrics[0], average_metrics[0]], [values[0], avg_values[0]])
    plt.xlabel('Score')
    plt.title('RMSE, Average RMSE')

    plt.subplot(3, 1, 2)
    plt.barh(metrics[1:], values[1:])
    plt.xlabel('Score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('Precision, Recall, Accuracy For Current Recommendation')

    plt.subplot(3, 1, 3)
    plt.barh(average_metrics[1:], avg_values[1:])
    plt.xlabel('Score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('Average Precision, Recall, Accuracy')

    plt.tight_layout()

    plt.savefig('plot.png')
    plt.show()
