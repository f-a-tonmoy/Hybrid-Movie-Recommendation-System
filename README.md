## 📝 Project Overview

The **Movie Recommendation System** predicts and recommends movies to users based on their preferences and viewing history. This system leverages collaborative filtering, content-based filtering, and user-specific evaluation metrics to deliver personalized recommendations. The application includes an interactive graphical user interface (GUI) for a seamless user experience.

---

## 📂 Project Structure

```plaintext
├── mrs.py                          # Core recommendation system logic
├── ui.py                           # User interface implementation (Tkinter)
├── setup.ipynb                     # Data preprocessing and setup notebook
├── Files used in preprocessing/    # Additional datasets and setup files
│   ├── movies.csv                  # Movie metadata
│   ├── overviews.csv               # Movie overview details
│   ├── ratings.csv                 # User ratings dataset
│   ├── tags.csv                    # User-generated tags for movies
│   └── system.ipynb                # System-wide setup notebook
├── eval_metrics.csv                # Evaluation metrics for the models (generated)
├── filtered_ratings.csv            # Preprocessed ratings for recommendation (generated)
├── movie_db.csv                    # Movie database used in recommendations (generated)
├── user_item_matrix.csv            # User-item matrix for collaborative filtering (generated)
├── logo.png                        # Application logo
├── plot.png                        # Evaluation and comparison plot
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
```

---

## 🚀 Features

1. **Core Recommendation Algorithms**:
   - **Collaborative Filtering**:
     - Utilizes Pearson correlation to find similar users and recommend movies.
   - **Content-Based Filtering**:
     - Jaccard similarity to recommend movies based on content and tags.
   - **Hybrid Approach**:
     - Combines collaborative and content-based filtering for better accuracy and tackcle cold start problem.

2. **Interactive User Interface**:
   - GUI built with **Tkinter**:
     - User login and validation system.
     - Search functionality for movies by title or genre.
     - Displays personalized movie recommendations.
     - Allows users to rate movies, update ratings, and get new recommendations.

3. **Evaluation and Feedback**:
   - Tracks model performance using metrics like RMSE, precision, recall, and accuracy.
   - Visualizes evaluation results in **plot.png**.

4. **Preprocessing**:
   - Created a user-item matrix from `ratings.csv` for collaborative filtering.
   - Cleans datasets and applies normalization for better model performance.

5. **Movie Similarity Recommendations**:
   - Suggests movies similar to the user's input based on overview and genre similarity.

---

## 📂 Dataset Details

The **MovieLens dataset** serves as the foundation for this project. It is a widely-used benchmark dataset for building recommendation systems, provided by [GroupLens Research](https://grouplens.org/datasets/movielens/). The dataset contains user ratings, movie metadata, and user-item interactions.

### Key Files:
1. **Ratings Dataset (`ratings.csv`)**:
   - Original dataset from MovieLens.
   - Includes:
     - `userId`: Unique ID for each user.
     - `movieId`: Unique ID for each movie.
     - `rating`: Rating given by the user (e.g., 1–5).
     - `timestamp`: When the rating was provided.

2. **User-Item Matrix (`user_item_matrix.csv`)**:
   - Captures user-movie interactions for collaborative filtering.
   - **Created During Preprocessing**: Generated in `setup.ipynb` from `filtered_ratings.csv`.

3. **Movie Database (`movie_db.csv`)**:
   - Contains metadata such as movie titles, genres, and average ratings.
   - **Created During Preprocessing**: Extracted and refined using `setup.ipynb`.

4. **Filtered Ratings (`filtered_ratings.csv`)**:
   - Preprocessed version of `ratings.csv` optimized for faster computations.
   - **Created During Preprocessing**: Produced using `setup.ipynb`.

5. **Evaluation Metrics (`eval_metrics.csv`)**:
   - Tracks performance metrics like RMSE, precision, recall, and accuracy.

### Dataset Source:
The **MovieLens dataset** was sourced from the official [MovieLens website](https://grouplens.org/datasets/movielens/). This raw dataset is cleaned and processed using `setup.ipynb` to generate the intermediate files required for the recommendation engine.

---

## 🛠️ Dependencies

Install the required libraries using:
```bash
pip install -r requirements.txt
```

### `requirements.txt` Content:
```plaintext
pandas
numpy
matplotlib
tkinter
```

---

## 🔍 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/f-a-tonmoy/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess data and set up the environment:
   ```bash
   jupyter notebook setup.ipynb
   ```

4. Run the application:
   ```bash
   python ui.py
   ```

---

## 🧪 Results

 **Hybrid Recommendations**:
  - Combines both methods for improved results, reducing bias from either approach.
  - Diverse recommendations.
  - Tackles cold start problem.
  - Real-time recommendations.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or create pull requests.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💬 Contact

For inquiries or feedback:
- **Fahim Ahamed (Tonmoy)**: [f.a.tonmoy00@gmail.com](mailto:f.a.tonmoy00@gmail.com)
- GitHub: [f-a-tonmoy](https://github.com/f-a-tonmoy)
```
