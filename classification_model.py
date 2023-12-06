import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'spotify_songs.csv'  # Replace with your file path
spotify_data = pd.read_csv(file_path)

# Dropping columns that are not needed for the model
spotify_data = spotify_data.drop(columns=['track_id', 'track_name', 'track_artist', 'track_album_id',
                                          'track_album_name', 'track_album_release_date',
                                          'playlist_name', 'playlist_id'])

# Handling missing values
spotify_data = spotify_data.dropna()

# Identifying categorical and numeric columns
categorical_cols = spotify_data.select_dtypes(include=['object', 'category']).columns
numeric_cols = spotify_data.select_dtypes(include=['int64', 'float64']).columns

# Encoding categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    if col != 'playlist_genre':  # Exclude the target variable from encoding
        spotify_data[col] = label_encoder.fit_transform(spotify_data[col])

# Encoding the target variable 'playlist_genre'
spotify_data['playlist_genre'] = label_encoder.fit_transform(spotify_data['playlist_genre'])

# Splitting the dataset into features (X) and target variable (y)
X = spotify_data.drop('playlist_genre', axis=1)
y = spotify_data['playlist_genre']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing only numeric features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Initialize the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train_scaled, y_train)

# Predicting the Test set results
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
