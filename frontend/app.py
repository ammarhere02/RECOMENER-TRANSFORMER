"""
Netflix Recommendation System - Flask Backend
This Flask application provides an API for the Netflix recommendation engine.
It uses Sentence Transformers for semantic similarity and KNN for finding similar content.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and data
df = None
embeddings = None
knn = None
model = None

def initialize_model():
    """
    Initialize the recommendation system.
    Loads the dataset, generates embeddings, and fits the KNN model.
    """
    global df, embeddings, knn, model

    print("🔄 Loading Netflix dataset...")
    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'netflix_titles.csv')
    df = pd.read_csv(csv_path)

    # Clean data - fill missing descriptions
    df['description'] = df['description'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')

    # Create combined text for better embeddings
    df['combined_text'] = df['description'] + ' ' + df['listed_in']

    print("🤖 Loading Sentence Transformer model...")
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("📊 Generating embeddings (this may take a moment)...")
    # Generate embeddings
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

    print("🔍 Fitting KNN model...")
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(embeddings)

    print("✅ Model initialized successfully!")

def get_recommendations(title, n_recommendations=5):
    """
    Get movie/TV show recommendations based on title.

    Args:
        title (str): The title to search for
        n_recommendations (int): Number of recommendations to return

    Returns:
        dict: Contains searched movie info and list of recommendations
    """
    global df, embeddings, knn

    # Find the movie by title (case-insensitive)
    mask = df['title'].str.lower() == title.lower()

    if not mask.any():
        # Try partial match
        mask = df['title'].str.lower().str.contains(title.lower(), na=False)

    if not mask.any():
        return None

    # Get the first matching movie
    idx = df[mask].index[0]
    movie_data = df.loc[idx]

    # Find similar movies using KNN
    distances, indices = knn.kneighbors([embeddings[idx]], n_neighbors=n_recommendations + 1)

    # Get recommendations (excluding the searched movie itself)
    recommendations = []
    for i, rec_idx in enumerate(indices[0][1:]):  # Skip first as it's the movie itself
        rec = df.iloc[rec_idx]
        recommendations.append({
            'title': rec['title'],
            'type': rec['type'],
            'year': int(rec['release_year']) if pd.notna(rec['release_year']) else None,
            'rating': rec['rating'] if pd.notna(rec['rating']) else 'Not Rated',
            'description': rec['description'][:200] + '...' if len(str(rec['description'])) > 200 else rec['description'],
            'similarity_score': round(1 - distances[0][i + 1], 3)
        })

    return {
        'searched': {
            'title': movie_data['title'],
            'type': movie_data['type'],
            'year': int(movie_data['release_year']) if pd.notna(movie_data['release_year']) else None,
            'rating': movie_data['rating'] if pd.notna(movie_data['rating']) else 'Not Rated',
            'description': movie_data['description'],
            'director': movie_data['director'] if pd.notna(movie_data['director']) else 'Unknown',
            'cast': movie_data['cast'] if pd.notna(movie_data['cast']) else 'Unknown'
        },
        'recommendations': recommendations
    }

# Routes
@app.route('/')
def serve_frontend():
    """Serve the frontend HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend():
    """
    API endpoint for getting recommendations.

    Query Parameters or JSON Body:
        title (str): The movie/show title to get recommendations for
        n (int, optional): Number of recommendations (default: 5)

    Returns:
        JSON response with searched movie and recommendations
    """
    if request.method == 'POST':
        data = request.get_json()
        title = data.get('title', '')
        n = data.get('n', 5)
    else:
        title = request.args.get('title', '')
        n = int(request.args.get('n', 5))

    if not title:
        return jsonify({'error': 'Title is required'}), 400

    result = get_recommendations(title, n)

    if result is None:
        return jsonify({'error': 'Title not found', 'query': title}), 404

    return jsonify(result)

@app.route('/api/search', methods=['GET'])
def search():
    """
    Search for titles matching a query.

    Query Parameters:
        q (str): Search query
        limit (int, optional): Maximum results (default: 10)

    Returns:
        JSON list of matching titles
    """
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # Search for matching titles
    mask = df['title'].str.lower().str.contains(query.lower(), na=False)
    matches = df[mask][['title', 'type', 'release_year', 'rating']].head(limit)

    results = matches.to_dict('records')
    return jsonify({'results': results, 'count': len(results)})

@app.route('/api/popular', methods=['GET'])
def popular():
    """
    Get popular/random titles for homepage display.

    Query Parameters:
        limit (int, optional): Number of titles (default: 10)

    Returns:
        JSON list of popular titles
    """
    limit = int(request.args.get('limit', 10))

    # Get random sample of titles
    sample = df.sample(n=min(limit, len(df)))[['title', 'type', 'release_year', 'rating', 'description']]
    sample['description'] = sample['description'].apply(
        lambda x: x[:150] + '...' if len(str(x)) > 150 else x
    )

    return jsonify({'titles': sample.to_dict('records')})

@app.route('/api/stats', methods=['GET'])
def stats():
    """
    Get dataset statistics.

    Returns:
        JSON with various statistics about the Netflix dataset
    """
    return jsonify({
        'total_titles': len(df),
        'movies': len(df[df['type'] == 'Movie']),
        'tv_shows': len(df[df['type'] == 'TV Show']),
        'countries': df['country'].nunique(),
        'years_range': {
            'min': int(df['release_year'].min()),
            'max': int(df['release_year'].max())
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🎬 Netflix Recommendation System")
    print("=" * 50)

    # Initialize the model
    initialize_model()

    print("\n🚀 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("=" * 50)

    # Run the Flask app
    app.run(debug=True, port=5000)

