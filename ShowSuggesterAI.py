import time
import logging
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def find_most_similar_show(input_string, shows_list):
    closest_match, _ = process.extractOne(input_string, shows_list)
    return closest_match



# Set up root logger to a higher level to avoid external libraries' logs
logging.basicConfig(level=logging.WARNING)
# Create a logger for your application
logger = logging.getLogger("ShowSuggesterAI")
logger.setLevel(logging.INFO)

def calculate_average_embedding(show_titles, embeddings_dict):
    embeddings = []
    for title in show_titles:
        if title in embeddings_dict:
            embeddings.append(embeddings_dict[title])
        else:
            # Use logger to log the message instead of print
            logger.info(f"Embedding not found for show: {title}")

    if embeddings:
        # Calculate the average embedding
        average_embedding = np.mean(embeddings, axis=0)
        return average_embedding
    else:
        return None


def generate_recommendations(average_embedding, favorite_shows, embeddings_dict, top_n=5):

    recommendations = []

    for show, embedding in embeddings_dict.items():
        if show not in favorite_shows:
            similarity = cosine_similarity([average_embedding], [embedding])[0][0]
            recommendations.append((show, similarity))

        # Sort by similarity score in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Adjust scores for the 2nd to 5th recommendations
    adjusted_recommendations = recommendations[:1]  # Keep the top recommendation as is
    for show, similarity in recommendations[1:top_n]:
        adjusted_similarity = similarity * (0.95 - 0.05 * (len(adjusted_recommendations) - 1))
        adjusted_recommendations.append((show, adjusted_similarity))

    # Convert similarity scores to percentages
    top_recommendations = [(show, round(similarity * 100, 2)) for show, similarity in adjusted_recommendations]

    return top_recommendations





def calculate_average_vector(show_titles, show_vectors_df):

    # Convert DataFrame to dictionary
    vectors_dict = show_vectors_df.T.to_dict('list')
    # Now you can use the existing logic to calculate the average vector
    return calculate_average_embedding(show_titles, vectors_dict)


def load_tv_show_vectors(csv_path):

    return pd.read_csv(csv_path, index_col='TV Show')




def generate_knn_recommendations(average_vector, favorite_shows, show_vectors_df, top_n=5):

    # Exclude favorite shows from the comparison
    comparison_df = show_vectors_df.drop(favorite_shows, errors='ignore')

    # Fit the KNN model
    knn = NearestNeighbors(n_neighbors=top_n, algorithm='auto').fit(comparison_df.values)

    # Find the k-nearest neighbors to the average vector
    distances, indices = knn.kneighbors([average_vector])

    # Get the titles of the closest shows
    closest_shows = comparison_df.iloc[indices[0]].index.tolist()
    return closest_shows





def main():

    # Load the serialized dictionary:
    # Replace the path with a relative path or a general placeholder.
    # For example: './data/embeddings_dict.pkl'
    with open('embeddings_dict.pkl', 'rb') as file:
        loaded_embeddings = pickle.load(file)
        # Assume 'loaded_embeddings' is your dictionary loaded from the pickle file

    while True:
        # Step 1: Ask the user for their favorite TV shows
        user_input = input(
            "Which TV shows did you love watching? Separate them by a comma.\nMake sure to enter more than 1 show: ")
        favorite_shows = [show.strip() for show in user_input.split(',')]

        # Load embeddings (assuming this is already done before this function)
        # with open('embeddings_dict.pkl', 'rb') as file:
        #     loaded_embeddings = pickle.load(file)

        # Step 2: Confirm the TV show names using fuzzy matching
        confirmed_shows = []
        for show in favorite_shows:
            closest_match = find_most_similar_show(show, list(loaded_embeddings.keys()))
            confirmed_shows.append(closest_match)

        confirmation = input(f"Just to make sure, do you mean {', '.join(confirmed_shows)}? (y/n) ")
        if confirmation.lower() == 'y':
            break
        else:
            print("Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly.")

        # Step 3: Generate recommendations
    print("Great! Generating recommendations...")
    average_embedding = calculate_average_embedding(confirmed_shows, loaded_embeddings)
    recommendations = generate_recommendations(average_embedding, confirmed_shows, loaded_embeddings)

    # Replace this path with a relative path or a general placeholder.
    # For example: './data/tvShowsFeatures.csv'
    csv_path = 'Replace me with the path ot tvShowsFeatures.csv in your system'
    show_vectors_df = load_tv_show_vectors(csv_path)

    average_vector = calculate_average_vector(confirmed_shows, show_vectors_df)

    knn_recommendations = generate_knn_recommendations(average_vector, favorite_shows, show_vectors_df)

    # Filter out duplicate recommendations
    recommended_shows = set([show for show, _ in recommendations])
    unique_knn_recommendations = [show for show in knn_recommendations if show not in recommended_shows]

    # Step 4: Display recommendations
    print("Here are the TV shows that I think you would love:")
    print("")
    for i, (show, similarity) in enumerate(recommendations[:5], start=1):
        print(f"{i}. {show} ({similarity}%)")
    print("")
    time.sleep(5)
    print("")
    # Step 3.5: Generate recommendations using the KNN algorithm


    print("--------------------------- beta algorithm ---------------------------")
    print("And here are some more recommended shows based on my newest beta algorithm. Hope you like them even better:")

    print("")  # Print the KNN-based recommendations nicely
    for i, show in enumerate(unique_knn_recommendations[:5], start=1):
        print(f"{i}. {show}")



if __name__ == "__main__":
    main()


