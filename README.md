<br/>
<p align="center">
  <h3 align="center">Revolutionizing Playlist Creation: Harness the Power of AI to Classify Music Genres Effortlessly</h3>

  <p align="center">
    Transform Your Music Experience - Where Algorithms Meet Harmonies!
    <br/>
    <br/>
  </p>
</p>



## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Problem Scenario

**Background**: A leading music streaming company, *Streamify*, is looking to enhance its playlist generation algorithms. Their objective is to automatically classify songs into appropriate genres based on their attributes. This improvement is expected to enhance user experience by providing more accurate and diverse playlist recommendations.

**Challenge**: Streamify has a vast and diverse collection of songs, each with multiple attributes like rhythm, tempo, artist influence, etc. Manually categorizing these songs into genres is labor-intensive and prone to errors. Additionally, genres can be subjective and overlap, making the classification task more complex.

**Data Availability**: The company has a substantial dataset (`spotify_songs.csv`) containing various attributes of songs, including track IDs, names, artist details, album information, and playlist genres.

**Objective**: To develop a machine learning model that can accurately classify songs into predefined genres based on their characteristics.

### Solution Implementation

1. **Data Preprocessing**:
   - Load the dataset and remove irrelevant columns that don't contribute to genre classification, such as track IDs and album names.
   - Handle missing values to ensure data quality.
   - Identify and separate categorical and numeric columns.

2. **Feature Engineering**:
   - Apply Label Encoding to transform categorical variables into a machine-readable form while ensuring the target variable 'playlist_genre' is also encoded.
   - Split the dataset into features (X) and the target (y).

3. **Data Splitting**:
   - Divide the data into training and testing sets to both train the model and evaluate its performance on unseen data.

4. **Data Normalization**:
   - Use StandardScaler to normalize numeric features, ensuring that all features contribute equally to the model's decision-making process.

5. **Model Selection and Training**:
   - Choose RandomForestClassifier for its robustness and ability to handle complex classification tasks.
   - Train the model on the processed training data.

6. **Model Evaluation**:
   - Predict genres on the test set and evaluate the model using accuracy and a detailed classification report, which includes precision, recall, and F1-scores for each genre.

7. **Performance Metrics**:
   - The accuracy score provides an overall effectiveness measure of the model.
   - The classification report offers insights into the model's performance across different genres, highlighting potential areas for improvement.

### Expected Outcome

The RandomForestClassifier is expected to accurately predict the genre of songs in the dataset. High accuracy and balanced precision and recall scores across all genres would indicate a successful model. This automated classification system can significantly streamline Streamify's playlist generation process, leading to more personalized and engaging user experiences.

### Further Considerations

- **Model Improvement**: Experiment with different models, feature selection techniques, and hyperparameter tuning for optimal performance.
- **User Feedback Incorporation**: Integrate user feedback to continuously refine the model, accommodating evolving music trends and preferences.
- **Scalability and Maintenance**: Ensure the model scales with the growing dataset and is regularly updated to reflect changes in music genres and user tastes.

The code is a typical machine learning pipeline using Python's `pandas`, `sklearn`, and a RandomForest classifier. It processes a dataset of Spotify songs to predict the genre of a playlist based on the characteristics of the songs it contains. 

1. **Importing Libraries**: 
   - `pandas` is used for data manipulation and analysis.
   - `sklearn` (Scikit-learn) provides tools for data preprocessing and machine learning models.

2. **Loading the Dataset**: 
   - The dataset `spotify_songs.csv` is loaded into a pandas DataFrame. This file should contain Spotify song data.

3. **Data Preprocessing**:
   - **Dropping Columns**: Unnecessary columns (like track IDs, names, album information, playlist information) are removed. These are likely non-numeric and not useful for the machine learning model.
   - **Handling Missing Values**: Rows with missing data are removed to maintain data integrity.
   - **Identifying Column Types**: Columns are categorized into categorical (non-numeric) and numeric for appropriate preprocessing.
   - **Encoding Categorical Variables**: Non-numeric categorical variables are encoded to numeric values using Label Encoding, except for the target variable 'playlist_genre'.
   - **Encoding Target Variable**: The target variable 'playlist_genre' is also label-encoded.

4. **Feature-Target Split**: 
   - The dataset is split into features (`X`) and the target variable (`y`).

5. **Training-Testing Split**: 
   - The data is split into training and testing sets to evaluate the model's performance. The `train_test_split` function is used with a test size of 20% and a random state for reproducibility.

6. **Feature Scaling**: 
   - Numeric features are scaled using StandardScaler to normalize their range. This is important for models like Random Forest, although it's generally more critical for models sensitive to feature scaling like SVM or kNN.

7. **Random Forest Classifier**: 
   - A RandomForest classifier is initialized and trained on the scaled training data. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputs the class that is the mode of the classes of individual trees.

8. **Model Training and Prediction**: 
   - The model is trained using the `fit` method on the training data.
   - It then predicts genres for the test set.

9. **Model Evaluation**: 
   - The model's performance is evaluated using accuracy and a classification report, which includes precision, recall, and F1-score for each class.

The script is comprehensive, covering key steps in a machine learning workflow: data preprocessing, feature engineering, model training, and evaluation. However, some improvements or additional steps could include exploratory data analysis, feature selection, hyperparameter tuning, and cross-validation to enhance model performance and reliability.

The output results from the classification model reveal several important metrics related to its performance in classifying Spotify playlist genres. Let's break down each component of the output:

1. **Accuracy (0.946246383432313)**: 
   - This figure (approximately 94.62%) represents the overall accuracy of your model. It indicates the proportion of total predictions that the model got right. In your case, the model correctly predicted the genre of approximately 94.62% of the playlists in the test set.

2. **Classification Report**: 
   - This report provides a more detailed analysis of the model's performance, broken down by each class (genre) in the target variable.

3. **Metrics per Class**:
   - **Precision**: Indicates the accuracy of positive predictions for each genre. For instance, a precision of 0.93 for genre 0 means that 93% of the playlists predicted as genre 0 were actually genre 0.
   - **Recall**: Measures the model's ability to detect a particular genre. For example, a recall of 0.92 for genre 0 means that the model correctly identified 92% of all actual genre 0 playlists.
   - **F1-Score**: Harmonic mean of precision and recall. It provides a single score that balances both the concerns of precision and recall in one number. An F1-score close to 1 indicates a very good balance between precision and recall.
   - **Support**: The number of actual occurrences of the class in the specified dataset. For example, there are 1218 instances of genre 0 in your test dataset.

4. **Overall Metrics**:
   - **Macro Average**: Averages the metrics (precision, recall, F1-score) for each class, treating all classes equally. This means it calculates the metric independently for each class and then takes the average (hence treating all classes equally).
   - **Weighted Average**: Similar to the macro average, but it accounts for the imbalance in the dataset by weighting the score of each class by its presence in the dataset.

5. **Interpretation**:
   - Your model shows high precision and recall across all genres, indicating it performs well in both correctly identifying each genre (precision) and in capturing a high proportion of actual instances of each genre (recall).
   - The high F1-scores across all genres suggest a good balance between precision and recall.
   - The overall accuracy of 94.62% is impressive, but it's important to consider the context and the distribution of classes in the dataset. If some genres are underrepresented, the model might still be biased towards more common genres.

6. **Exit Code (0)**:
   - This indicates that your Python script ran successfully without any errors.

In summary, these results suggest that the RandomForest classifier performs very effectively in classifying Spotify playlist genres based on the features provided. However, it's crucial to consider the dataset's balance and the real-world applicability of the model. For instance, if new genres emerge or if the characteristics of genres change over time, the model may need retraining or updating.

## Built With

This section outlines the tools and libraries used in the development of the Spotify Genre Classification project. The project leverages several powerful Python libraries, each contributing significantly to various aspects of the machine learning pipeline, from data preprocessing to model training and evaluation.

1. **Pandas (pandas)**: 
   - **Purpose**: Data manipulation and analysis.
   - **Key Features Used**: 
     - Reading CSV files (`read_csv`).
     - DataFrame operations for data cleaning and preprocessing (e.g., `drop`, `dropna`).
   - **Version**: [Specify version if known, e.g., 1.3.4].
   - **Website**: [Pandas Official Website](https://pandas.pydata.org/).

2. **Scikit-learn (sklearn)**: 
   - **Purpose**: Machine learning and predictive data analysis.
   - **Key Features Used**: 
     - Data splitting (`train_test_split`).
     - Preprocessing tools (`StandardScaler`, `LabelEncoder`).
     - Random Forest algorithm implementation (`RandomForestClassifier`).
     - Performance metrics (`classification_report`, `accuracy_score`).
   - **Version**: [Specify version, e.g., 0.24.2].
   - **Website**: [Scikit-learn Official Website](https://scikit-learn.org/stable/).

3. **Random Forest Classifier**:
   - **Purpose**: Core machine learning algorithm for classification.
   - **Key Features Used**:
     - Ability to handle both categorical and numerical data.
     - Robustness against overfitting.
     - Ensemble learning method for improved accuracy.
   - **Integrated within Scikit-learn**.

4. **Label Encoding**:
   - **Purpose**: Convert categorical text data into a model-understandable numerical format.
   - **Key Features Used**:
     - Encoding labels with value between 0 and n_classes-1.
   - **Integrated within Scikit-learn**.

5. **StandardScaler**:
   - **Purpose**: Standardize features by removing the mean and scaling to unit variance.
   - **Key Features Used**:
     - Scaling of dataset to normalize the distribution.
   - **Integrated within Scikit-learn**.

6. **Python**:
   - **Purpose**: Primary programming language.
   - **Key Features Used**: 
     - Versatility and ease of use for data science and machine learning.
   - **Version**: [Specify version, e.g., Python 3.8].
   - **Website**: [Python Official Website](https://www.python.org/).

### Additional Tools (If Any)

- **Version Control (e.g., Git/GitHub)**: For code management and collaboration.
- **Development Environment (e.g., PyCharm, Jupyter Notebook)**: For writing and testing code.

### Installation and Dependencies

To replicate or contribute to this project, ensure Python is installed along with the aforementioned libraries. Specific version requirements and installation commands (e.g., `pip install pandas==1.3.4 sklearn`) should be listed to maintain consistency across different development environments.

## Getting Started

This section guides you through setting up the Spotify Genre Classification project on your local machine for development and testing purposes. Follow these instructions to get a copy of the project up and running.

#### Prerequisites

Before you begin, ensure you have the following installed:
1. **Python**: The project is written in Python. [Download and install Python](https://www.python.org/downloads/), preferably the latest version.
2. **Pandas and Scikit-learn**: These Python libraries are essential for data manipulation and machine learning functionalities. They can be installed via pip:
   ```
   pip install pandas scikit-learn
   ```

#### Installation

1. **Clone the Repository**:
   - Use Git or checkout with SVN using the web URL to clone the repository to your local machine.
   ```
   git clone [repository URL]
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   - Navigate to the project directory in your terminal and set up a Python virtual environment. This keeps dependencies required by the project separate from your global Python installation.
     ```
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows: `venv\Scripts\activate`
     - On Unix or MacOS: `source venv/bin/activate`

3. **Install Dependencies**:
   - Ensure all required packages are installed by running:
     ```
     pip install -r requirements.txt
     ```

#### Running the Project

1. **Prepare Your Data**:
   - Ensure you have the `spotify_songs.csv` dataset. Replace `file_path` in the script with the path to your CSV file.

2. **Run the Script**:
   - Execute the Python script to start the genre classification process.
     ```
     python classification_model.py
     ```

3. **Interpret the Output**:
   - After running the script, you will see the model's accuracy and a detailed classification report in the console. This will indicate how well the model is performing.

#### Testing

- Conduct tests to validate the functionality. You can create test cases to check if the data preprocessing, model training, and predictions are working as expected.

#### Contributing

If you'd like to contribute to the project, please read `CONTRIBUTING.md` (if available) for details on our code of conduct, and the process for submitting pull requests.

#### Support

For any questions or issues, feel free to open an issue in the repository or contact the repository maintainers.

---

By following these steps, you should be able to successfully set up and run the Spotify Genre Classification project. The instructions assume a basic familiarity with Python programming and command-line operations.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/ShaanCoding/Automating-Music-Categorization-With-Machine-Learning/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/ShaanCoding/Automating-Music-Categorization-With-Machine-Learning/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com/ShaanCoding/Automating-Music-Categorization-With-Machine-Learning/blob/main/LICENSE.md) for more information.

## Authors

* **Robbie** - *PhD Computer Science Student* - [Robbie](https://github.com/TribeOfJudahLion) - **

## Acknowledgements

* []()
* []()
* []()
