# Netflix Movies & TV Shows Analysis & Recommendation System

A comprehensive data analysis and machine learning project that explores Netflix's content library and builds a semantic-based recommendation engine.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-NLP-red.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Analysis Highlights](#analysis-highlights)
- [Recommendation Engine](#recommendation-engine)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project performs in-depth analysis of Netflix's content catalog and builds an intelligent movie/TV show recommendation system using semantic text embeddings. The project covers:

1. **Data Preprocessing & Cleaning** - Handling missing values, duplicates, and data standardization
2. **Exploratory Data Analysis** - Analyzing genres, directors, countries, and release trends
3. **Data Visualization** - Creating insightful charts and heatmaps
4. **Machine Learning Model** - Rating prediction using Random Forest
5. **Semantic Recommendation Engine** - Content-based recommendations using NLP

## ✨ Features

- 📊 **Comprehensive EDA**: Explore Netflix's content distribution across genres, countries, and time
- 📈 **Interactive Visualizations**: Beautiful charts using Matplotlib and Seaborn
- 🤖 **ML-based Rating Prediction**: Predict content ratings using Random Forest Regressor
- 🎬 **Smart Recommendations**: Get personalized movie/TV show recommendations based on semantic similarity
- 🔍 **Semantic Search**: Uses Sentence Transformers for understanding content descriptions

## 📁 Dataset

The project uses the `netflix_titles.csv` dataset containing:
- **show_id**: Unique identifier
- **type**: Movie or TV Show
- **title**: Title of the content
- **director**: Director name(s)
- **cast**: Cast members
- **country**: Country of production
- **date_added**: Date added to Netflix
- **release_year**: Year of release
- **rating**: Content rating (PG, R, TV-MA, etc.)
- **duration**: Length in minutes or seasons
- **listed_in**: Genres/Categories
- **description**: Content description

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd IDS-PROJECT
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Jupyter Notebook**
```bash
jupyter notebook "IDS Project.ipynb"
```

## 📂 Project Structure

```
IDS-PROJECT/
├── IDS Project.ipynb      # Main Jupyter notebook with all analysis
├── netflix_titles.csv     # Dataset
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── frontend/              # Web interface for recommendations
│   ├── index.html         # Frontend HTML
│   └── app.py             # Flask backend (optional)
└── images/                # Visualization exports
```

## 📊 Analysis Highlights

### Top 10 Genres
- International Movies
- Dramas
- Comedies
- International TV Shows
- Documentaries
- Action & Adventure
- TV Dramas
- Independent Movies
- Children & Family Movies
- Romantic Movies

### Top Content Producing Countries
1. 🇺🇸 United States
2. 🇮🇳 India
3. 🇬🇧 United Kingdom
4. 🇯🇵 Japan
5. 🇰🇷 South Korea

### Content Growth
- Netflix content additions peaked around **2019-2020**
- Steady growth in content library over the years

## 🎬 Recommendation Engine

The recommendation system uses **Semantic Text Analysis** with the following approach:

### How It Works

1. **Text Preprocessing**: Combines description and genre information
2. **Embedding Generation**: Uses `all-MiniLM-L6-v2` Sentence Transformer model
3. **Similarity Matching**: K-Nearest Neighbors with cosine similarity
4. **Recommendations**: Returns top 5 similar titles based on content

### Example Usage

```python
# Get recommendations for a movie
recommend_movies("Narcos", df, embeddings, knn)

# Output:
# These are your recommendations, according to your search
# ['El Chapo' 'Fearless' 'Drug Lords' 'The Mechanism' 'ZeroZeroZero']
```

## 💻 Usage

### Running the Analysis

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Load data
df = pd.read_csv("netflix_titles.csv")

# Preprocess and generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["description"].tolist())

# Build recommendation engine
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(embeddings)

# Get recommendations
recommend_movies("Your Movie Title", df, embeddings, knn)
```

### Using the Web Interface

```bash
cd frontend
python app.py
# Open http://localhost:5000 in your browser
```

## 📈 Results

### Rating Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Duration, Year Added, Country
- **RMSE**: ~1.13
- **R²**: ~0.02

> **Note**: The low R² indicates that content ratings are not strongly predicted by these features. This is expected as ratings are subjective and content-dependent.

### Recommendation Engine
- **Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Similarity Metric**: Cosine Similarity
- **Performance**: High-quality semantic recommendations based on content descriptions

## 🛠 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, Random Forest |
| **NLP/Embeddings** | Sentence Transformers, Hugging Face |
| **Similarity Search** | K-Nearest Neighbors |
| **Development** | Jupyter Notebook, Python 3.9+ |

## 📝 Key Insights

1. **Genre Distribution**: International content dominates Netflix's library
2. **Geographic Focus**: US leads in content production, followed by India and UK
3. **Content Growth**: Significant growth in content additions through 2019-2020
4. **Rating Patterns**: TV-MA and TV-14 are the most common ratings
5. **Recommendation Quality**: Semantic embeddings provide meaningful content-based recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Ammar Khan**

---

⭐ If you found this project helpful, please consider giving it a star!

