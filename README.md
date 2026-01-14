# Movies & TV Shows Analysis & Recommendation System

A comprehensive data analysis and machine learning project that explores Netflix's content library and builds a semantic-based recommendation engine using **Sentence Transformers**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-NLP-red.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Challenges](#dataset-challenges)
- [Why Transformers](#why-transformers)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Analysis Highlights](#analysis-highlights)
- [Recommendation Engine](#recommendation-engine)
- [Web Interface](#web-interface)
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

## ⚠️ Dataset Challenges & Flaws

During our analysis, we discovered several **critical issues** with the Netflix dataset that significantly impacted traditional ML approaches:

### 1. **Missing Values**
| Column | Missing % | Impact |
|--------|-----------|--------|
| `director` | ~30% | Cannot reliably use director as a feature |
| `cast` | ~10% | Incomplete cast information |
| `country` | ~10% | Geographic analysis affected |
| `date_added` | ~1% | Timeline gaps |

### 2. **Data Quality Issues**
- **Multi-valued columns**: `country`, `listed_in`, and `cast` contain multiple comma-separated values
- **Inconsistent formatting**: Different date formats, inconsistent naming conventions
- **Null values causing errors**: Operations like `explode()` failed on NaN values
- **Unhashable types**: List columns caused `TypeError: unhashable type: 'list'` during visualization

### 3. **Rating Prediction Failure**
We attempted to predict content ratings using Random Forest Regressor with numerical features:

```
RMSE: 1.131643175154974
R²: 0.020475435983366763
```

**Why did it fail?**
- **R² = 0.02** means the model explains only **2%** of the variance
- Rating is a **categorical/subjective** attribute, not suitable for regression
- Available numerical features (duration, year) have **weak correlation** with ratings
- Missing features like viewer demographics, reviews, and engagement metrics

### 4. **Visualization Errors**
```python
# This caused errors due to NaN and list values:
sns.barplot(x=country_counts.values, y=country_counts.index, palette='coolwarm')
# TypeError: unhashable type: 'list'
```

The multi-valued columns required `explode()` operations, but NaN values caused failures.

## 🚀 Why Transformers? The Solution

Given the dataset limitations, **traditional ML approaches failed** to provide accurate predictions or recommendations. We pivoted to a **Transformer-based semantic approach**:

### Traditional ML vs Transformer Approach

| Aspect | Traditional ML | Transformer (Our Solution) |
|--------|---------------|---------------------------|
| **Features** | Numerical only | Text (descriptions, genres) |
| **Missing Data** | Causes failures | Handles gracefully |
| **Understanding** | Surface-level patterns | Deep semantic meaning |
| **Similarity** | Feature distance | Content meaning similarity |
| **Accuracy** | Poor (R² = 0.02) | High-quality recommendations |

### How Transformers Solve the Problem

1. **Semantic Understanding**: Instead of relying on flawed numerical features, we use **text descriptions** which are complete for all entries
2. **Dense Embeddings**: `all-MiniLM-L6-v2` model converts descriptions into 384-dimensional vectors capturing meaning
3. **Cosine Similarity**: Finds truly similar content based on plot, theme, and genre
4. **Robust to Missing Data**: Works with description + genre, ignoring incomplete columns

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

## 🌐 Web Interface

A minimal Flask-based web interface is provided for easy interaction with the recommendation system.

### Features
- 🔍 **Search**: Find movies/TV shows by title
- 🎬 **Recommendations**: Get 5 similar titles with similarity scores
- 📊 **Statistics**: View dataset stats (total titles, movies vs TV shows, etc.)
- 🎲 **Popular**: Browse random popular titles

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend HTML |
| `/api/recommend?title=<title>` | GET/POST | Get recommendations |
| `/api/search?q=<query>` | GET | Search titles |
| `/api/popular?limit=10` | GET | Get random titles |
| `/api/stats` | GET | Dataset statistics |

### Running the Server
```bash
cd frontend
pip install flask flask-cors sentence-transformers
python app.py
```

## 📈 Results

### ❌ Rating Prediction Model (Failed Approach)
- **Algorithm**: Random Forest Regressor
- **Features**: Duration, Year Added, Country (encoded)
- **RMSE**: 1.13
- **R²**: 0.02 (Only 2% variance explained)

**Why It Failed:**
- Rating is a **categorical/subjective** attribute - not suitable for regression
- Numerical features have **no meaningful correlation** with content ratings
- Dataset lacks crucial features like user reviews, view counts, engagement metrics
- This is **NOT a classification problem** - ratings like "TV-MA", "PG-13" are categories, not continuous values

> **Lesson Learned**: The dataset's numerical features cannot predict ratings. We needed a different approach.

### ✅ Transformer-based Recommendation Engine (Success)
- **Model**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Embedding Dimension**: 384
- **Similarity Metric**: Cosine Similarity with KNN
- **Performance**: **High-quality semantic recommendations**

**Why It Works:**
- Uses complete `description` field (no missing values issue)
- Captures **semantic meaning** of content
- Finds truly similar movies based on plot and themes
- Robust to dataset flaws

### Example Output
```
Search: "Narcos"
Recommendations:
1. El Chapo (Similarity: 0.89)
2. Drug Lords (Similarity: 0.85)
3. The Mechanism (Similarity: 0.82)
4. ZeroZeroZero (Similarity: 0.79)
5. Fearless (Similarity: 0.76)
```

## 🛠 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, Random Forest |
| **NLP/Embeddings** | Sentence Transformers, Hugging Face |
| **Similarity Search** | K-Nearest Neighbors (Cosine) |
| **Web Framework** | Flask, Flask-CORS |
| **Development** | Jupyter Notebook, Python 3.9+ |

## 📝 Key Insights & Lessons Learned

### Dataset Insights
1. **Genre Distribution**: International content dominates Netflix's library
2. **Geographic Focus**: US leads in content production, followed by India and UK
3. **Content Growth**: Significant growth in content additions through 2019-2020
4. **Rating Patterns**: TV-MA and TV-14 are the most common ratings

### Technical Lessons
5. **ML Limitation**: Traditional ML fails when features don't correlate with target
6. **Data Quality Matters**: Missing values and multi-valued columns require careful handling
7. **Right Tool for the Job**: Transformers excel at semantic understanding where numerical features fail
8. **Recommendation Quality**: Semantic embeddings provide meaningful content-based recommendations

### What We Learned About the Dataset Flaws
- ❌ **Rating prediction is NOT feasible** with available features
- ❌ **Multi-valued columns** need `explode()` but fail on NaN
- ❌ **Categorical encoding** of countries/genres loses semantic meaning
- ✅ **Text descriptions** are the most reliable and complete feature
- ✅ **Transformer embeddings** capture what numerical features cannot

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Ammar Khan**

---

⭐ If you found this project helpful, please consider giving it a star!

