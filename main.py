import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import kagglehub
import os


class NFLPredictor:
    def __init__(self):
        self.model = None
        self.median_probability = None

    def load_kaggle_data(self):
        """Load data from Kaggle"""
        try:
            # Download the specific spreadsheet we want
            path = kagglehub.dataset_download("tobycrabtree/nfl-scores-and-betting-data", "spreadspoke_scores.csv")

            print(f"Loading data from: {path}")

            # Read the CSV file
            df = pd.read_csv(path)

            # Debug information
            print("\nDataset loaded successfully")
            print(f"Number of rows: {len(df)}")
            print("\nColumns in dataset:")
            print(df.columns.tolist())

            return df

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def make_prediction_interactive(self):
        """Interactive method to get game predictions"""
        print("\nEnter game details:")
        try:
            favorite_spread = float(input("Spread for favorite team (e.g. -3.5): "))
            over_under = float(input("Over/Under line (e.g. 44.5): "))
            temp = float(input("Temperature in Fahrenheit (enter 70 if indoor): "))
            wind = float(input("Wind speed in MPH (enter 0 if indoor): "))
            humidity = float(input("Humidity percentage (enter 40 if indoor): "))
            date = input("Game date (MM/DD/YYYY): ")

            prediction = self.predict_game(
                favorite_spread=favorite_spread,
                over_under=over_under,
                temp=temp,
                wind_speed=wind,
                humidity=humidity,
                date=date
            )

            print("\nPrediction Results:")
            print(f"Probability of Over: {prediction['probability']:.3f}")
            print(f"Prediction: {prediction['prediction']}")
            if prediction['confidence']:
                print(f"Confidence: {prediction['confidence']:.1%}")
            else:
                print("Confidence: No strong prediction for this scenario")

            # Additional context
            factors = []
            if temp < 40:
                factors.append("Cold weather tends to reduce scoring")
            if wind > 15:
                factors.append("High winds typically favor the Under")
            if over_under > 50:
                factors.append("High over/under lines tend to go Under more often")

            if factors:
                print("\nKey Factors:")
                for factor in factors:
                    print(f"- {factor}")

        except ValueError as e:
            print(f"\nError: Please enter valid numbers. {str(e)}")
            return

        return prediction

    def prepare_data(self, df):
        """Prepare data for modeling"""
        # Convert date to month
        df['month'] = pd.to_datetime(df['schedule_date']).dt.month

        # Define features
        features = ['spread_favorite', 'over_under_line', 'temperature',
                    'wind', 'humidity', 'month']

        # Rename columns to match our expected names
        df = df.rename(columns={
            'weather_wind_mph': 'wind',
            'weather_temperature': 'temperature',
            'weather_humidity': 'humidity'
        })

        # Convert over_under_line to numeric, removing any non-numeric characters
        df['over_under_line'] = pd.to_numeric(df['over_under_line'], errors='coerce')

        # Convert spread_favorite to numeric as well
        df['spread_favorite'] = pd.to_numeric(df['spread_favorite'], errors='coerce')

        # Handle stadium defaults (using stadium_neutral as proxy for dome)
        df.loc[df['stadium_neutral'] == 1, 'temperature'] = 70
        df.loc[df['stadium_neutral'] == 1, 'wind'] = 0
        df.loc[df['stadium_neutral'] == 1, 'humidity'] = 40

        # Fill missing weather data with defaults
        df['temperature'] = df['temperature'].fillna(70)
        df['wind'] = df['wind'].fillna(0)
        df['humidity'] = df['humidity'].fillna(40)

        # Create target (1 for over, 0 for under)
        df['target'] = (df['score_home'].astype(float) + df['score_away'].astype(float) > df['over_under_line']).astype(
            int)

        # Remove rows with any remaining null values in our feature set
        df = df.dropna(subset=features + ['target'])

        return df[features], df['target']

    def train(self):
        """Train the model"""
        # Load and prepare data
        df = self.load_kaggle_data()

        # Debug: Print column names and first few rows
        print("\nAvailable columns in dataset:")
        print(df.columns.tolist())
        print("\nFirst few rows of data:")
        print(df.head())

        # Look for season column (it might be named differently)
        season_columns = [col for col in df.columns if 'season' in col.lower()]
        print("\nPossible season columns:", season_columns)

        if not season_columns:
            raise ValueError("Could not find season column in dataset")

        # Use the first found season column
        season_column = season_columns[0]
        print(f"\nUsing column '{season_column}' for season filtering")

        # Filter data from 2002 onwards
        df = df[df[season_column] >= 2002].copy()

        # Prepare features and target
        X, y = self.prepare_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=42)

        # Create and train model
        self.model = GradientBoostingClassifier(
            n_estimators=1099,
            learning_rate=0.001,
            max_depth=2,
            min_samples_leaf=10,
            subsample=0.5,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Calculate median probability for threshold
        train_probs = self.model.predict_proba(X_train)[:, 1]
        self.median_probability = np.median(train_probs)

        # Calculate accuracy
        test_probs = self.model.predict_proba(X_test)[:, 1]
        predictions = (test_probs > self.median_probability).astype(int)
        accuracy = accuracy_score(y_test, predictions)

        # Print feature importances
        feature_importance = pd.DataFrame({
            'feature': ['spread_favorite', 'over_under_line', 'temperature',
                        'wind', 'humidity', 'month'],
            'importance': self.model.feature_importances_
        })
        print("\nFeature Importances:")
        print(feature_importance.sort_values('importance', ascending=False))

        return accuracy

    def predict_game(self, favorite_spread, over_under, temp, wind_speed, humidity, date):
        """Make prediction for a single game"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")

        # Create feature array with names
        month = pd.to_datetime(date).month
        features = pd.DataFrame([[favorite_spread, over_under, temp, wind_speed, humidity, month]],
                                columns=['spread_favorite', 'over_under_line', 'temperature',
                                         'wind', 'humidity', 'month'])

        # Get probability
        prob = self.model.predict_proba(features)[0, 1]

        # Determine if we should make a prediction based on month and probability
        prediction = None
        confidence = None

        if month == 9:  # September
            if 0.440 <= prob <= 0.495:
                prediction = "Under"
                confidence = 0.623
        elif month == 10:  # October
            if 0.452 <= prob <= 0.469:
                prediction = "Under"
                confidence = 0.762
        elif month == 11:  # November
            if 0.432 <= prob <= 0.45:
                prediction = "Under"
                confidence = 0.571
            elif 0.511 <= prob <= 0.537:
                prediction = "Over"
                confidence = 0.571
        elif month == 12:  # December
            prediction = "Over" if prob > 0.507 else "Under"
            confidence = 0.639
        elif month == 1:  # January
            prediction = "Over" if prob > 0.507 else "Under"
            confidence = 0.627
        elif month == 2:  # February
            if abs(prob - 0.498) < 0.001:
                prediction = "Under"
                confidence = 0.612

        return {
            'probability': prob,
            'prediction': prediction,
            'confidence': confidence
        }


if __name__ == "__main__":
    predictor = NFLPredictor()

    print("Training model...")
    accuracy = predictor.train()
    print(f"\nModel accuracy: {accuracy:.3f}")

    while True:
        choice = input("\nWould you like to make a prediction? (y/n): ")
        if choice.lower() != 'y':
            break
        predictor.make_prediction_interactive()