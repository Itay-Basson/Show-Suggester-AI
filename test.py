from ShowSuggesterAI import find_most_similar_show, calculate_average_embedding, generate_recommendations
import numpy as np


# Updated mock data
mock_shows_list = ["Breaking Bad", "Game of Thrones", "Friends", "The Office", "The Witcher"]
mock_embeddings_dict = {
    "Breaking Bad": np.array([0.1, 0.2, 0.3]),
    "Game of Thrones": np.array([0.2, 0.3, 0.4]),
    "Friends": np.array([0.3, 0.4, 0.5]),
    "The Office": np.array([0.4, 0.5, 0.6]),
    "The Witcher": np.array([0.5, 0.6, 0.7])
}
mock_favorite_shows = ["Breaking Bad", "Friends"]

def test_find_most_similar_show():
    # Test for exact match
    assert find_most_similar_show("Breaking Bad", mock_shows_list) == "Breaking Bad"
    # Test for partial match
    assert find_most_similar_show("Breaking", mock_shows_list) == "Breaking Bad"


def test_calculate_average_embedding():
    average_embedding = calculate_average_embedding(["Breaking Bad", "Friends"], mock_embeddings_dict)
    expected_average = np.mean([mock_embeddings_dict["Breaking Bad"], mock_embeddings_dict["Friends"]], axis=0)
    # Check if the average is calculated correctly
    assert np.array_equal(average_embedding, expected_average)


def test_generate_recommendations():
    average_embedding = np.array([0.2, 0.3, 0.4])
    recommendations = generate_recommendations(average_embedding, mock_favorite_shows, mock_embeddings_dict)
    # Test number of recommendations
    assert len(recommendations) <= 5 and len(recommendations) > 0
    # Test if favorite shows are not in recommendations
    assert all(show not in mock_favorite_shows for show, _ in recommendations)
    # Test if recommendations are sorted by similarity
    similarities = [similarity for _, similarity in recommendations]
    assert all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1))
