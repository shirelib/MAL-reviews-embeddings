print("start")

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import timeit
from langdetect import detect
from sklearn.metrics.pairwise import pairwise_distances



###############################------   Setup & Pre-processing   ------###############################
df = pd.read_csv('reviews.csv')
print("reviews file loaded")

def is_numeric(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False

# Filtering out non-numeric values (for id)
df = df[df['uid'].apply(is_numeric)]
df = df.dropna(how='all').reset_index()

# Only long reviews in english
df['review_length'] = df['text'].apply(lambda x: len(x.split()))
df = df[df['review_length'] >= 50]
df = df[df['text'].apply(lambda x: detect(x) == 'en')]

# Using only some lines for faster testing
df = df.head(15)

# Json in scores column ==> to separate columns
score_cols = df['scores'].apply(eval).apply(pd.Series)
score_cols = score_cols.applymap(pd.to_numeric, errors='coerce')

# adding score columns to original df
df = pd.concat([df, score_cols], axis=1)



###############################------   TFIDF   ------###############################

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df['text']).toarray()
vocab = tfidf.vocabulary_
reverse_vocab = {v: k for k, v in vocab.items()}
feature_names = tfidf.get_feature_names_out()
df_tfidf = pd.DataFrame(X_tfidf, columns=feature_names)
idx = X_tfidf.argsort(axis=1) # a NumPy array that contains the indices for sorting each row in ascending order
tfidf_max10 = idx[:, -70:] # max tfidf words

df_tfidf['top50'] = [[reverse_vocab.get(item) for item in row] for row in tfidf_max10]
df_tfidf = df_tfidf.reset_index() # check without reset_index()

df = pd.concat([df.reset_index(), df_tfidf['top50']], axis=1)



###############################------   Embedding   ------###############################
print("EMBEDDINGS")
# DistilRoberta
drmodel = AutoModel.from_pretrained("distilroberta-base")
drtokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
max_length = 85 # of tokens (was 60 when num max was 50)
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = drtokenizer(text, return_tensors="pt",
                         max_length=max_length, padding="max_length", truncation=True)
    # Remove stopwords and apply lemmatization
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stopwords]
    # Combine the tokens back into a text
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def embedrob(txt):
    preprocessed_text = preprocess_text(txt)

    inputs = drtokenizer(preprocessed_text, return_tensors="pt",
                         max_length=max_length, padding="max_length", truncation=True)
    start_time = timeit.default_timer()

    outputs = drmodel(**inputs)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print("Elapsed time!: {:.2f} seconds".format(elapsed_time))
    return outputs.last_hidden_state # added .last_hidden_state

embeddings = df['top50'].apply(" ".join).apply(embedrob)
df['embeddings'] = embeddings


###############################------   PCA   ------###############################

df['embeddings'] = df['embeddings'].apply(lambda array: array[0].flatten())
embeddings_array = np.stack(df['embeddings'].apply(lambda x: x.detach().numpy()))

# Perform PCA on the embeddings matrix
n_comp = 6
pca = PCA(n_components=n_comp)  # the number of components to reduce to
embeddings_pca = pca.fit_transform(embeddings_array)

# Normalization
scaler = StandardScaler()
normalized_embeddings_pca = scaler.fit_transform(embeddings_pca)

# Create a new DataFrame with the PCA results
pca_df = pd.DataFrame(normalized_embeddings_pca, columns=[f'PC{i+1}' for i in range(n_comp)])
df = pd.concat([df, pca_df], axis=1)

# Display the resulting DataFrame with PCA results
print("\n\n", df)
#df.to_csv('out4.csv')


# Group by anime and aggregate the embeddings
    # aggregation after pca: combines different opinions into one.

grouped_embeddings = df.groupby('anime_uid')[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
                                              'score', 'Overall', 'Story', 'Animation',
                                              'Sound', 'Character', 'Enjoyment']].mean()
df = grouped_embeddings.reset_index()

#grouped_embeddings.to_csv('grouped_out4.csv')


###############################------   Merging with anime title and info   ------###############################

animedf = pd.read_csv('animes.csv')
print("animes file loaded")
animedf = animedf.rename(columns={'uid': 'anime_uid'})  # or add axis=1 instead of columns

# remove duplicates
animedf.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

df = df.merge(animedf, on='anime_uid', how='left')


###############################------   Visualizations   ------###############################

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['PC1'], df['PC2'])
print("df.PC1", df.PC1)
print("df.PC2", df.PC2)

# Add labels for each point
for i, row in df.iterrows():
    plt.annotate(row['title'], (row['PC1'], row['PC2']))

# Add axis labels and title
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Visualization of Anime Data')

# Show the plot
print("opening plot")
plt.show()

###############################------   Cosine Similarity   ------###############################

# Drop any non-numeric columns if necessary
cosdf = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]

# Calculate cosine similarities from cosine distances by subtracting the result from 1
cosine_sim = 1 - pairwise_distances(cosdf, metric='cosine')

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(cosdf)

# avoiding similarity of a vector with itself not being exactly 1:
np.fill_diagonal(cosine_sim, 1.0)

# The result is a square matrix where cosine_sim[i][j] represents the cosine similarity between observations i and j

def cosine_single_anime(indx, mat):
    print("Input index:", indx)
    print("Name of the show:", df['title'].iloc[indx])
    cosine_specific_obs = mat[indx].copy()
    # avoiding top similarity being with itself:
    cosine_specific_obs[cosine_specific_obs == 1.0] = 0
    most_similar_index = np.argmax(abs(cosine_specific_obs))
    most_similar_cosine = cosine_specific_obs[most_similar_index]
    print("Most similar show:\n", df['title'].iloc[most_similar_index])
    print("Corresponding cosine:", most_similar_cosine, "\n")
    return df.iloc[most_similar_index]

def cosine_top3_anime(indx, mat):
    print("Input index:", indx)
    print("Name of the show:", df['title'].iloc[indx])
    cosine_specific_obs = mat[indx].copy()
    # avoiding top similarity being with itself:
    cosine_specific_obs[cosine_specific_obs == 1.0] = 0
    most_similar_indices = np.argpartition(abs(cosine_specific_obs), -3)[-3:]
    most_similar_indices = most_similar_indices[::-1] # reverse order
    most_similar_cosine = cosine_specific_obs[most_similar_indices]
    print("Top 3 similar shows:\n", df['title'].iloc[most_similar_indices])
    print("Corresponding cosines:", most_similar_cosine, "\n")
    return df.iloc[most_similar_indices]

cosine_single_anime(2, cosine_sim)
cosine_single_anime(3, cosine_sim)
cosine_single_anime(5, cosine_sim)

# find anime index by name
abyss_indx = df.index[df['title'] == "Made in Abyss"][0]
print("Made in Abyss index", abyss_indx)

cosine_single_anime(abyss_indx, cosine_sim)
cosine_top3_anime(abyss_indx, cosine_sim)





