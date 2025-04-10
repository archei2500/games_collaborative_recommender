import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import the recommender classes
from recommender import CollaborativeRecommender
from train import preprocess_data, fit, save_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test recommendation system')
    parser.add_argument('--data_path', type=str, default=None, help='Path to custom dataset CSV file')
    parser.add_argument('--model_path', type=str, default='models/saved_model/',
                        help='Directory to save/load trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of test split (0.1-0.5)')
    parser.add_argument('--n_components', type=str, default='auto', help='Number of SVD components')
    return parser.parse_args()


def generate_test_data(num_users=1000, num_games=200, sparsity=0.9):
    """Generate synthetic test data with controlled sparsity"""
    np.random.seed(42)

    # Create user-game matrix
    data = []
    for user_id in range(num_users):
        # Each user plays a random subset of games
        games_played = np.random.choice(
            num_games,
            size=int(num_games * (1 - sparsity)),
            replace=False
        )
        for game_id in games_played:
            playtime = np.random.lognormal(mean=3, sigma=1)
            data.append({
                'steamid': f"765611980000{1000 + user_id}",
                'appid': game_id + 1,  # avoid 0 as game ID
                'playtime_forever': playtime
            })

    return pd.DataFrame(data)


# либо берёт данные из пути, либо генерит свои данные (сомнительно)
def load_or_generate_data(data_path):
    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}")
        data = pd.read_csv(data_path)

        # проверка столбцов
        required_columns = {'steamid', 'appid', 'playtime_forever'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")

        return data
    else:
        print("No dataset provided, generating synthetic data")
        return generate_test_data()


def train_test_split_data(data, test_size=0.2):
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        stratify=data['steamid']  # Ensure users appear in both sets
    )
    return train_data, test_data


def test_recommendation_scenarios(recommender, test_data):
    # # Get a known user from test data
    # test_users = test_data['steamid'].unique()
    # test_user_id = test_users[0] if len(test_users) > 0 else "7656119800001000"

    # # Get some game IDs from test data
    # test_games = test_data['appid'].unique()[:2] if len(test_data) > 0 else [450, 1330]
    # test_games_with_time = [(test_games[0], 10), (test_games[1], 5)]

    # Find a user with multiple games in test data
    user_game_counts = test_data['steamid'].value_counts()
    multi_game_users = user_game_counts[user_game_counts > 1].index

    if len(multi_game_users) > 0:
        test_user_id = multi_game_users[0]
        user_games = test_data[test_data['steamid'] == test_user_id]['appid'].tolist()

        # Split user's games into "known" (in train) and "new" (to test)
        test_games = user_games[:2]
        test_games_with_time = [(user_games[0], 10), (user_games[1], 5)]  # With playtime
    else:
        # Fallback if no users with multiple games found
        test_user_id = "7656119800001000"
        test_games = [450, 1330]
        test_games_with_time = [(450, 10), (1330, 5)]

    print("\nTesting recommendation scenarios:")

    # 1. Test cold start
    cold_start_recs = recommender.recommend()
    assert len(cold_start_recs) > 0, "Cold start should return recommendations"
    print(f"1. Cold start recommendations: {cold_start_recs}")

    # 2. Test with game list
    game_list_recs = recommender.recommend(user_games=test_games)
    assert len(game_list_recs) > 0, "Should return recommendations for game list"
    print(f"2. Game list recommendations: {game_list_recs}")

    # 3. Test with games and playtimes
    game_time_recs = recommender.recommend(user_games=test_games_with_time)
    assert len(game_time_recs) > 0, "Should return recommendations for games with playtime"
    print(f"3. Games with playtime recommendations: {game_time_recs}")

    # 4. Test with known user
    user_recs = recommender.recommend(user_id=test_user_id)
    assert len(user_recs) > 0, "Should return recommendations for known user"
    print(f"4. User-specific recommendations: {user_recs[:5]}")

    # 5. Test with unknown user
    unknown_user_recs = recommender.recommend(user_id="9999999999999999")
    assert len(unknown_user_recs) > 0, "Should fall back to other methods for unknown user"
    print(f"5. Unknown user recommendations: {unknown_user_recs}...")

    # 6. Known user and their new games
    if len(multi_game_users) > 0:
        new_games_recs = recommender.recommend(
            user_id=test_user_id,
            user_games=user_games
        )
        assert len(new_games_recs) > 0, "Should handle known user with new games"
        print(f"6. Known user with new games recommendations: {new_games_recs}")
    else:
        print("6. Skipping known user with new games test - no suitable users found")


def evaluate_model(recommender, test_data):
    test_users = test_data['steamid'].unique()
    test_user_mapping = {user: idx for idx, user in enumerate(test_users)}

    # Create test user-item matrix
    test_matrix = test_data.pivot_table(
        index='steamid',
        columns='appid',
        values='playtime_log',
        fill_value=0
    )

    # Выравнивание по столбцам обучающей матрицы
    test_matrix = test_matrix.reindex(columns=recommender.user_item_matrix.columns, fill_value=0)

    # Calculate RMSE for known users
    known_users = [user for user in test_users if str(user) in recommender.user_mapping]
    if not known_users:
        return float('nan'), float('nan')

    # Get predictions for known users
    predictions = []
    actuals = []

    for user in known_users:
        user_idx = recommender.user_mapping[str(user)]
        test_idx = test_user_mapping[user]

        # Get actual playtimes
        actual = test_matrix.iloc[test_idx].values
        actuals.extend(actual)

        # Get predicted scores
        pred = recommender.user_embeddings[user_idx] @ recommender.item_embeddings
        predictions.extend(pred)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # Calculate coverage (percentage of user-item pairs we can predict for)
    coverage = len(known_users) / len(test_users)

    return rmse, coverage


def run_tests(args):
    # Load or generate data
    data = load_or_generate_data(args.data_path)
    data = preprocess_data(data)

    # Split data
    train_data, test_data = train_test_split_data(data, test_size=args.test_size)

    # train recommender
    print("\nTraining model...")
    fit(train_data, random_state=42)
    print("\nLoading trained model...")
    recommender = CollaborativeRecommender()

    # Test recommendation scenarios
    test_recommendation_scenarios(recommender, test_data)

    # Evaluate model performance
    print("\nEvaluating model performance...")
    rmse, coverage = evaluate_model(recommender, test_data)
    print(f"Model RMSE: {rmse:.4f}")
    print(f"User Coverage: {coverage:.2%}")

    # Basic quality checks
    assert rmse < 1.0, "RMSE should be reasonable (model should learn something)"
    assert coverage > 0.5, "Model should cover majority of test users"

    # Test model persistence
    print("\nTesting model persistence...")
    save_model(model_path=args.model_path,
               svd=recommender.svd,
               user_embeddings=recommender.user_embeddings,
               item_embeddings=recommender.item_embeddings,
               user_mapping=recommender.user_mapping,
               game_mapping=recommender.game_mapping,
               user_item_matrix=recommender.user_item_matrix,
               game_popularity=recommender.game_popularity)
    loaded_recommender = CollaborativeRecommender(model_path=args.model_path)
    assert loaded_recommender.is_trained, "Loaded model should be marked as trained"
    print("Model successfully saved and loaded")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    run_tests(args)
