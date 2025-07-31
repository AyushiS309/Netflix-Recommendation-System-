import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

warnings.filterwarnings('ignore')

# -----------------------------
# ✅ 1️⃣ Streamlit config + custom CSS
# -----------------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }

    .stButton > button {
        background: linear-gradient(135deg, #E50914 0%, #B20710 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(229, 9, 20, 0.4);
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #c62828;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ✅ 2️⃣ Load & clean data with better error handling
# -----------------------------
@st.cache_data
def load_and_clean_data():
    """Load and clean Netflix dataset with comprehensive error handling."""
    try:
        # Check if file exists
        if not os.path.exists('netflix_titles.csv'):
            st.error("""
            ❌ **Dataset not found!**  
            
            Please ensure `netflix_titles.csv` is in the same folder as this app.  
            
            **Download the dataset:**
            - [Kaggle Netflix Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
            - Place the downloaded CSV file in the same directory as this app
            """)
            return pd.DataFrame()
        
        # Load the dataset
        df = pd.read_csv('netflix_titles.csv')
        
        # Show initial data info
        st.sidebar.success(f"✅ Successfully loaded: {len(df)} titles")
        
        # Display column information for debugging
        with st.expander("📊 Dataset Information", expanded=False):
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {list(df.columns)}")
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
            st.write("**Sample Data:**")
            st.dataframe(df.head())
        
        # Clean and handle missing values
        required_columns = ['title', 'type', 'release_year']
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            st.error(f"❌ Missing required columns: {missing_required}")
            return pd.DataFrame()
        
        # Fill missing values with appropriate defaults
        df['director'] = df['director'].fillna('Unknown Director')
        df['cast'] = df['cast'].fillna('Unknown Cast')
        df['country'] = df['country'].fillna('Unknown Country')
        df['description'] = df['description'].fillna('No description available')
        df['rating'] = df['rating'].fillna('Not Rated')
        df['listed_in'] = df['listed_in'].fillna('General')
        df['duration'] = df['duration'].fillna('Unknown Duration')
        
        # Handle date conversion
        if 'date_added' in df.columns:
            df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        
        # Clean release year
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
        
        # Remove rows with missing titles (critical)
        initial_count = len(df)
        df = df.dropna(subset=['title'])
        if len(df) < initial_count:
            st.warning(f"⚠️ Removed {initial_count - len(df)} rows with missing titles")
        
        # Create numeric rating for sorting (improved logic)
        def extract_duration_minutes(duration_str, content_type):
            """Extract duration in minutes for movies, or assign random score for TV shows."""
            try:
                if content_type == 'Movie' and isinstance(duration_str, str):
                    if 'min' in duration_str:
                        return float(duration_str.split(' ')[0])
                    else:
                        return np.random.uniform(90, 180)  # Default movie length
                else:
                    # For TV shows, create a score based on seasons
                    if isinstance(duration_str, str) and 'Season' in duration_str:
                        seasons = duration_str.split(' ')[0]
                        try:
                            return float(seasons) * 10  # More seasons = higher score
                        except:
                            return np.random.uniform(20, 80)
                    else:
                        return np.random.uniform(20, 80)
            except:
                return np.random.uniform(50, 90)
        
        df['numeric_rating'] = df.apply(
            lambda x: extract_duration_minutes(x['duration'], x['type']), axis=1
        )
        
        # Create combined features for recommendation (THIS WAS THE MISSING PART!)
        def safe_combine_features(row):
            """Safely combine features for text analysis."""
            try:
                features = []
                
                # Add each feature if it exists and is not null
                if pd.notna(row.get('listed_in')):
                    features.append(str(row['listed_in']))
                
                if pd.notna(row.get('description')):
                    features.append(str(row['description']))
                    
                if pd.notna(row.get('cast')):
                    features.append(str(row['cast']))
                    
                if pd.notna(row.get('director')):
                    features.append(str(row['director']))
                
                if pd.notna(row.get('country')):
                    features.append(str(row['country']))
                
                return ' '.join(features)
            except:
                return 'Unknown Content'
        
        df['combined_features'] = df.apply(safe_combine_features, axis=1)
        
        # Validate the combined features
        if df['combined_features'].isnull().any():
            st.warning("⚠️ Some combined features are null, filling with default values")
            df['combined_features'] = df['combined_features'].fillna('Unknown Content')
        
        # Final validation
        st.sidebar.info(f"🔧 Data processed successfully!")
        st.sidebar.info(f"📝 Combined features created for {len(df)} titles")
        
        return df
        
    except FileNotFoundError:
        st.error("""
        ❌ **Dataset not found!**  
        
        Please place `netflix_titles.csv` in the same folder as this app.  
        [Download from Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
        """)
        return pd.DataFrame()
        
    except pd.errors.EmptyDataError:
        st.error("❌ The CSV file is empty or corrupted.")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ **Error loading dataset:** {str(e)}")
        st.info("💡 **Troubleshooting tips:**")
        st.info("- Check if the CSV file is not corrupted")
        st.info("- Ensure the file has the correct column names")
        st.info("- Try re-downloading the dataset")
        return pd.DataFrame()

# -----------------------------
# ✅ 3️⃣ Recommendation logic with better error handling
# -----------------------------
@st.cache_data
def create_recommendation_models(df):
    """Create TF-IDF vectorizer and cosine similarity matrix."""
    try:
        if df.empty:
            st.error("❌ Cannot create recommendation models: Dataset is empty")
            return None, None
            
        if 'combined_features' not in df.columns:
            st.error("❌ Cannot create recommendation models: 'combined_features' column missing")
            return None, None
        
        # Check if combined_features has valid data
        if df['combined_features'].isnull().all():
            st.error("❌ Cannot create recommendation models: All combined features are null")
            return None, None
        
        # Create TF-IDF vectorizer with better parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        # Fit and transform the combined features
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        st.sidebar.success(f"✅ Recommendation models created successfully!")
        st.sidebar.info(f"📊 TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return cosine_sim, tfidf
        
    except Exception as e:
        st.error(f"❌ Error creating recommendation models: {str(e)}")
        return None, None

def get_content_recommendations(title, df, cosine_sim, num_recommendations=5):
    """Get content-based recommendations for a given title."""
    try:
        if cosine_sim is None:
            return pd.DataFrame()
        
        # Find the index of the movie/show
        indices = df[df['title'].str.lower() == title.lower()].index
        
        if len(indices) == 0:
            return pd.DataFrame()
        
        idx = indices[0]
        
        # Get similarity scores for all movies/shows
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies/shows based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices of the most similar movies/shows (excluding the input movie)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        # Get the recommended movies/shows
        recommendations = df.iloc[movie_indices][[
            'title', 'listed_in', 'type', 'release_year', 'rating', 'description'
        ]].copy()
        
        # Add similarity scores
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()

def get_popular_recommendations(df, num_recommendations=5):
    """Get popular recommendations based on release year and rating."""
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Sort by release year and numeric rating
        popular = df.nlargest(num_recommendations, ['release_year', 'numeric_rating'])
        
        return popular[[
            'title', 'listed_in', 'type', 'release_year', 'rating', 'description'
        ]]
        
    except Exception as e:
        st.error(f"Error getting popular recommendations: {str(e)}")
        return pd.DataFrame()

def get_genre_recommendations(genre, df, num_recommendations=5):
    """Get recommendations based on selected genre."""
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Filter by genre
        genre_df = df[df['listed_in'].str.contains(genre, case=False, na=False)]
        
        if len(genre_df) == 0:
            return pd.DataFrame()
        
        # Get top recommendations from this genre
        recommendations = genre_df.nlargest(num_recommendations, ['release_year', 'numeric_rating'])
        
        return recommendations[[
            'title', 'listed_in', 'type', 'release_year', 'rating', 'description'
        ]]
        
    except Exception as e:
        st.error(f"Error getting genre recommendations: {str(e)}")
        return pd.DataFrame()

def display_recommendation_card(row):
    """Display a recommendation card with proper formatting."""
    try:
        # Safely get description
        description = str(row.get('description', 'No description available'))
        truncated_desc = description[:200] + '...' if len(description) > 200 else description
        
        # Safely get similarity score if it exists
        similarity_html = ""
        if 'similarity_score' in row:
            similarity_html = f"<p><strong>Similarity:</strong> {row['similarity_score']:.2%}</p>"
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>🎬 {row['title']}</h4>
            <p><strong>Type:</strong> {row['type']}</p>
            <p><strong>Genres:</strong> {row['listed_in']}</p>
            <p><strong>Year:</strong> {int(row['release_year']) if pd.notna(row['release_year']) else 'Unknown'}</p>
            <p><strong>Rating:</strong> {row['rating']}</p>
            {similarity_html}
            <p>{truncated_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying recommendation: {str(e)}")

# -----------------------------
# ✅ 4️⃣ Main application with improved error handling
# -----------------------------
def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🎬 Netflix Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite show or movie!</p>', unsafe_allow_html=True)
    
    # Load data with spinner
    with st.spinner('🔄 Loading Netflix data...'):
        df = load_and_clean_data()
    
    # Stop execution if data loading failed
    if df.empty:
        st.stop()
    
    # Create recommendation models
    with st.spinner('🤖 Building recommendation models...'):
        cosine_sim, tfidf = create_recommendation_models(df)
    
    # Stop if model creation failed
    if cosine_sim is None:
        st.error("❌ Failed to create recommendation models. Please check your data.")
        st.stop()
    
    # Sidebar options
    st.sidebar.markdown("## 🎯 Recommendation Options")
    option = st.sidebar.selectbox(
        "Choose recommendation type:",
        ["Content-Based", "Popular", "By Genre", "Data Analysis"]
    )
    
    # Content-Based Recommendations
    if option == "Content-Based":
        st.markdown("### 🔍 Content-Based Recommendations")
        st.markdown("Find similar content based on genres, cast, director, and description.")
        
        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input(
                "Enter a title:", 
                placeholder="e.g., Stranger Things, The Crown, Black Mirror"
            )
        
        with col2:
            num_recs = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Search suggestions
        if user_input:
            # Find partial matches
            matches = df[df['title'].str.lower().str.contains(user_input.lower(), na=False)]
            if len(matches) > 0 and len(matches) < 10:
                st.info(f"💡 Similar titles: {', '.join(matches['title'].head(5).tolist())}")
        
        if st.button("🎯 Get Recommendations", key="content_recs"):
            if user_input:
                with st.spinner("🔍 Finding similar content..."):
                    recs = get_content_recommendations(user_input, df, cosine_sim, num_recs)
                
                if not recs.empty:
                    st.success(f"✅ Found {len(recs)} recommendations for '{user_input}'")
                    for _, row in recs.iterrows():
                        display_recommendation_card(row)
                else:
                    st.warning(f"❌ No recommendations found for '{user_input}'. Please check the title and try again.")
                    
                    # Suggest similar titles
                    similar_titles = df[df['title'].str.lower().str.contains(
                        user_input.lower().split()[0] if user_input else '', na=False
                    )]['title'].head(5).tolist()
                    
                    if similar_titles:
                        st.info(f"💡 Try these titles: {', '.join(similar_titles)}")
            else:
                st.warning("⚠️ Please enter a title to get recommendations.")
    
    # Popular Recommendations
    elif option == "Popular":
        st.markdown("### 🔥 Popular Recommendations")
        st.markdown("Trending content based on release year and ratings.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            num_recs = st.slider("Number of recommendations:", 1, 15, 8)
        
        with col2:
            content_type = st.selectbox("Content Type:", ["All", "Movie", "TV Show"])
        
        if st.button("🔥 Get Popular Content", key="popular_recs"):
            with st.spinner("📈 Finding popular content..."):
                # Filter by content type if specified
                filtered_df = df if content_type == "All" else df[df['type'] == content_type]
                popular = get_popular_recommendations(filtered_df, num_recs)
            
            if not popular.empty:
                st.success(f"✅ Here are the top {len(popular)} popular {content_type.lower() if content_type != 'All' else 'titles'}")
                for _, row in popular.iterrows():
                    display_recommendation_card(row)
            else:
                st.warning("❌ No popular content found.")
    
    # Genre-Based Recommendations
    elif option == "By Genre":
        st.markdown("### 🎭 Genre-Based Recommendations")
        st.markdown("Discover content from your favorite genres.")
        
        # Extract all unique genres
        genres = set()
        for g in df['listed_in'].dropna():
            genres.update([x.strip() for x in str(g).split(',')])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_genre = st.selectbox("Pick a genre:", sorted(genres))
        
        with col2:
            num_recs = st.slider("Number of recommendations:", 1, 15, 8)
        
        # Show genre statistics
        if selected_genre:
            genre_count = len(df[df['listed_in'].str.contains(selected_genre, case=False, na=False)])
            st.info(f"📊 Found {genre_count} titles in '{selected_genre}' genre")
        
        if st.button("🎭 Get Genre Recommendations", key="genre_recs"):
            with st.spinner(f"🎬 Finding {selected_genre} content..."):
                genre_recs = get_genre_recommendations(selected_genre, df, num_recs)
            
            if not genre_recs.empty:
                st.success(f"✅ Top {len(genre_recs)} recommendations in '{selected_genre}' genre")
                for _, row in genre_recs.iterrows():
                    display_recommendation_card(row)
            else:
                st.warning(f"❌ No content found for genre: '{selected_genre}'")
    
    # Data Analysis
    elif option == "Data Analysis":
        st.markdown("### 📊 Netflix Data Analysis")
        st.markdown("Explore insights from the Netflix dataset.")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_titles = len(df)
            st.markdown(f"<div class='metric-card'><h3>{total_titles}</h3><p>Total Titles</p></div>", 
                       unsafe_allow_html=True)
        
        with col2:
            total_movies = len(df[df['type'] == 'Movie'])
            st.markdown(f"<div class='metric-card'><h3>{total_movies}</h3><p>Movies</p></div>", 
                       unsafe_allow_html=True)
        
        with col3:
            total_shows = len(df[df['type'] == 'TV Show'])
            st.markdown(f"<div class='metric-card'><h3>{total_shows}</h3><p>TV Shows</p></div>", 
                       unsafe_allow_html=True)
        
        with col4:
            avg_year = int(df['release_year'].mean()) if not df['release_year'].isna().all() else "N/A"
            st.markdown(f"<div class='metric-card'><h3>{avg_year}</h3><p>Avg Release Year</p></div>", 
                       unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Content type distribution
            type_counts = df['type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title='Content Type Distribution',
                color_discrete_map={'Movie': '#E50914', 'TV Show': '#B20710'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Top genres
            all_genres = []
            for genres in df['listed_in'].dropna():
                all_genres.extend([g.strip() for g in str(genres).split(',')])
            
            genre_counts = pd.Series(all_genres).value_counts().head(10)
            fig_genres = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title='Top 10 Genres',
                color=genre_counts.values,
                color_continuous_scale='Reds'
            )
            fig_genres.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_genres, use_container_width=True)
        
        # Release year distribution
        fig_hist = px.histogram(
            df, 
            x='release_year', 
            title='Content Release Year Distribution',
            color='type',
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#B20710'},
            nbins=50
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top countries
        top_countries = df['country'].value_counts().head(15)
        fig_countries = px.bar(
            x=top_countries.index,
            y=top_countries.values,
            title='Top 15 Countries by Content Production',
            color=top_countries.values,
            color_continuous_scale='Reds'
        )
        fig_countries.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_countries, use_container_width=True)
    
    # Footer with sample titles
    st.markdown("---")
    sample_titles = df['title'].sample(min(10, len(df))).tolist()
    st.markdown("**💡 Try searching for these titles:**")
    st.write(", ".join(sample_titles))
    
    # Additional info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with ❤️ using Streamlit • Data from Netflix • Updated 2024
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please refresh the page and try again.")