
# TV Show Recommendations System

## Description
The TV Show Recommendations System is a Python-based project that leverages natural language processing and machine learning techniques to recommend TV shows to users based on their preferences. Utilizing OpenAI's embeddings for semantic analysis and various similarity metrics, the system provides personalized show recommendations.

## Features
- **Semantic Analysis**: Generates embeddings for TV show descriptions to understand content at a semantic level.
- **Fuzzy Matching**: Enhances user input accuracy through fuzzy string matching, making the system more user-friendly and resilient to typos or partial names.
- **Personalized Recommendations**: Uses cosine similarity and K-Nearest Neighbors algorithms to recommend shows that closely match the user's preferences.
- **Interactive User Interface**: Engages users in a conversational manner to refine their preferences and deliver tailored recommendations.

## Getting Started

### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.6 or later installed
- Access to OpenAI API (requires an API key)
- The following Python packages installed: `openai`, `pandas`, `numpy`, `scikit-learn`, `fuzzywuzzy`, `pickle`

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/tv-show-recommender.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
To start using the TV Show Recommender System, follow these steps:
1. Ensure you have a CSV file with TV show titles and descriptions.
2. Run `createSerializedPickleFile.py` to preprocess and serialize TV show data:
   ```bash
   python createSerializedPickleFile.py
   ```
3. Launch the recommendation system interface:
   ```bash
   python ShowSuggesterAI.py
   ```
4. Follow the interactive prompts to input your favorite TV shows and receive recommendations.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- OpenAI for the API and embeddings
- The fuzzywuzzy library for string matching
- The scikit-learn team for machine learning tools
