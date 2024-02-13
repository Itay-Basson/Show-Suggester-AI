import os
import pandas as pd
import pickle
from openai import OpenAI


# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get('your-openai-api-key'))


def get_embeddings(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def read_csv_and_extract_columns(file_path, title_col='Title', description_col='Description'):
    try:
        df = pd.read_csv(file_path)
        extracted_df = df[[title_col, description_col]]
        return extracted_df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None


def save_embeddings_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Embeddings saved to {filename}")


def main():
    file_path = '<path_to_your_csv_file>'  # Replace <path_to_your_csv_file> with the actual path to your CSV file
    tv_shows_df = read_csv_and_extract_columns(file_path)

    if tv_shows_df is not None:
        embeddings_dict = {}
        for _, row in tv_shows_df.iterrows():
            title = row['Title']
            description = row['Description']
            embedding = get_embeddings(description)
            if embedding:
                embeddings_dict[title] = embedding

        # Print size of the dictionary before serialization


        print("First 3 shows and embeddings before serialization:")
        for title in list(embeddings_dict.keys())[:3]:
            print(title, embeddings_dict[title])

        save_embeddings_to_pickle(embeddings_dict, 'embeddings_dict.pkl')




if __name__ == "__main__":
    main()
