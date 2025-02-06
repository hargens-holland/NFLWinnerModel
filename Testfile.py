import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import kagglehub
import os
from datetime import datetime


class NFLPredictionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def download_dataset(self):
        """
        Download the dataset using kagglehub
        """
        try:
            path = kagglehub.dataset_download("tobycrabtree/nfl-scores-and-betting-data")
            csv_file = None
            for file in os.listdir(path):
                if file.endswith('.csv') and 'spreadspoke_scores' in file:
                    csv_file = os.path.join(path, file)
                    break

            print(f"Dataset downloaded successfully to: {csv_file}")
            return csv_file
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            raise

    def calculate_team_stats(self, df):
        """
        Calculate historical team statistics
        """
        stats = {}
        df = df.sort_values('schedule_date')

        for _, game in df.iterrows():
            home_team = game['team_home']
            away_team = game['team_away']

            for team in [home_team, away_team]:
                if team not in stats:
                    stats[team] = {
                        'last_5_games': [],
                        'season_wins': 0,
                        'season_losses': 0,
                        'home_wins': 0,
                        'away_wins': 0,
                        'points_scored': [],
                        'points_allowed': []
                    }

            home_win = game['score_home'] > game['score_away']

            if home_win:
                stats[home_team]['season_wins'] += 1
                stats[away_team]['season_losses'] += 1
                stats[home_team]['home_wins'] += 1
            else:
                stats[away_team]['season_wins'] += 1
                stats[home_team]['season_losses'] += 1
                stats[away_team]['away_wins'] += 1

            stats[home_team]['last_5_games'].append(1 if home_win else 0)
            stats[away_team]['last_5_games'].append(0 if home_win else 1)
            stats[home_team]['last_5_games'] = stats[home_team]['last_5_games'][-5:]
            stats[away_team]['last_5_games'] = stats[away_team]['last_5_games'][-5:]

            stats[home_team]['points_scored'].append(game['score_home'])
            stats[home_team]['points_allowed'].append(game['score_away'])
            stats[away_team]['points_scored'].append(game['score_away'])
            stats[away_team]['points_allowed'].append(game['score_home'])

            for team in [home_team, away_team]:
                stats[team]['points_scored'] = stats[team]['points_scored'][-16:]
                stats[team]['points_allowed'] = stats[team]['points_allowed'][-16:]

        return stats

    def prepare_data(self):
        """
        Load and prepare the dataset with pre-game features
        """
        csv_path = self.download_dataset()
        df = pd.read_csv(csv_path)

        print("Calculating historical team statistics...")
        team_stats = self.calculate_team_stats(df)

        features_list = []
        targets = []
        df = df.sort_values('schedule_date')

        for _, game in df.iterrows():
            home_team = game['team_home']
            away_team = game['team_away']

            if home_team in team_stats and away_team in team_stats:
                home_stats = team_stats[home_team]
                away_stats = team_stats[away_team]

                features = {
                    'home_team_win_pct': home_stats['season_wins'] / max(1, home_stats['season_wins'] + home_stats[
                        'season_losses']),
                    'away_team_win_pct': away_stats['season_wins'] / max(1, away_stats['season_wins'] + away_stats[
                        'season_losses']),
                    'home_team_home_win_pct': home_stats['home_wins'] / max(1, home_stats['season_wins'] + home_stats[
                        'season_losses']),
                    'away_team_away_win_pct': away_stats['away_wins'] / max(1, away_stats['season_wins'] + away_stats[
                        'season_losses']),
                    'home_team_last_5': sum(home_stats['last_5_games']) / len(home_stats['last_5_games']) if home_stats[
                        'last_5_games'] else 0,
                    'away_team_last_5': sum(away_stats['last_5_games']) / len(away_stats['last_5_games']) if away_stats[
                        'last_5_games'] else 0,
                    'home_team_avg_points': np.mean(home_stats['points_scored']) if home_stats['points_scored'] else 0,
                    'away_team_avg_points': np.mean(away_stats['points_scored']) if away_stats['points_scored'] else 0,
                    'home_team_avg_points_allowed': np.mean(home_stats['points_allowed']) if home_stats[
                        'points_allowed'] else 0,
                    'away_team_avg_points_allowed': np.mean(away_stats['points_allowed']) if away_stats[
                        'points_allowed'] else 0,
                    'month': pd.to_datetime(game['schedule_date']).month,
                    'weather': game['weather_detail'],
                    'stadium_neutral': game['stadium_neutral'],
                    'over_under_line': game['over_under_line'],
                    'spread_favorite': game['spread_favorite']
                }

                features_list.append(features)
                targets.append(1 if game['score_home'] > game['score_away'] else 0)

        X = pd.DataFrame(features_list)
        y = np.array(targets)

        numeric_features = [col for col in X.columns if col != 'weather']
        for col in numeric_features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        X['weather'] = X['weather'].fillna('unknown')

        print("\nFeatures being used:", X.columns.tolist())
        print(f"Dataset shape: {X.shape}")

        return X, y

    def build_model(self, input_shape):
        """
        Build an improved model architecture with regularization
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(32, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def train(self, X, y, epochs=100, batch_size=64, validation_split=0.2):
        """
        Train with improved parameters and early stopping
        """
        numeric_features = [col for col in X.columns if col != 'weather']
        categorical_features = ['weather']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_transformed = self.preprocessor.fit_transform(X)
        input_shape = X_transformed.shape[1]

        self.build_model(input_shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=y
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        class_weights = {0: 1.0, 1: 1.0}
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {cls: total / (2.0 * count) for cls, count in zip(unique, counts)}

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=1
        )

        test_results = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = (self.model.predict(X_test, verbose=0) >= 0.5).astype(int)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        return history, test_results

    def predict_game(self, game_data):
        """
        Improved prediction with probability thresholding
        """
        game_df = pd.DataFrame([game_data])
        X_game = self.preprocessor.transform(game_df)

        prediction = self.model.predict(X_game, verbose=0)[0][0]
        confidence = max(prediction, 1 - prediction)

        result = "Home Team Wins" if prediction >= 0.5 else "Away Team Wins"

        return result, prediction, confidence


def main():
    try:
        print("Initializing NFL Prediction Model...")
        nfl_model = NFLPredictionModel()

        print("Preparing data...")
        X, y = nfl_model.prepare_data()

        print("Training model...")
        history, test_results = nfl_model.train(X, y, epochs=50)

        print("\nPlease enter the following statistics for your game prediction:")

        example_game = {}

        print("\n--- Team Win Percentages ---")
        example_game['home_team_win_pct'] = float(input("Home team win percentage (0.0 to 1.0): "))
        example_game['away_team_win_pct'] = float(input("Away team win percentage (0.0 to 1.0): "))
        example_game['home_team_home_win_pct'] = float(input("Home team's win percentage at home (0.0 to 1.0): "))
        example_game['away_team_away_win_pct'] = float(input("Away team's win percentage on road (0.0 to 1.0): "))

        print("\n--- Recent Form ---")
        example_game['home_team_last_5'] = float(input("Home team's wins in last 5 games (0.0 to 1.0): "))
        example_game['away_team_last_5'] = float(input("Away team's wins in last 5 games (0.0 to 1.0): "))

        print("\n--- Scoring Trends ---")
        example_game['home_team_avg_points'] = float(input("Home team's average points per game: "))
        example_game['away_team_avg_points'] = float(input("Away team's average points per game: "))
        example_game['home_team_avg_points_allowed'] = float(input("Home team's average points allowed per game: "))
        example_game['away_team_avg_points_allowed'] = float(input("Away team's average points allowed per game: "))

        print("\n--- Game Context ---")
        example_game['month'] = int(input("Month of the game (1-12): "))
        example_game['weather'] = input("Weather conditions (clear/rain/snow/dome): ")
        example_game['stadium_neutral'] = int(input("Neutral stadium? (0 for no, 1 for yes): "))
        example_game['over_under_line'] = float(input("Over/under line from betting odds: "))
        example_game['spread_favorite'] = float(input("Spread for favorite team (negative number): "))

        result, probability, confidence = nfl_model.predict_game(example_game)
        print(f"\nFinal Prediction: {result}")
        print(f"Win Probability: {probability:.2%}")
        print(f"Confidence: {confidence:.2%}")

    except ValueError as ve:
        print(f"\nError: Please enter valid numerical values for all statistics.")
        print("Win percentages should be between 0 and 1 (e.g., 0.75 for 75%)")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()