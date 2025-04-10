import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import joblib
from collections import defaultdict
import os


def preprocess_data(data):
    # Удаление дубликатов
    clean_data = data.drop_duplicates(['steamid', 'appid'], keep='last')

    # Логарифмирование времени игры
    clean_data.loc[:, 'playtime_log'] = np.log1p(clean_data['playtime_forever'])

    return clean_data


def save_model(model_path, svd, user_embeddings, item_embeddings, user_mapping, game_mapping, user_item_matrix,
               game_popularity):
    # Создаем директорию, если её нет
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(svd, f'{model_path}svd_model.pkl')
    joblib.dump(user_embeddings, f'{model_path}user_embeddings.pkl')
    joblib.dump(item_embeddings, f'{model_path}item_embeddings.pkl')
    joblib.dump(user_mapping, f'{model_path}user_mapping.pkl')
    joblib.dump(game_mapping, f'{model_path}game_mapping.pkl')
    joblib.dump(user_item_matrix, f'{model_path}user_item_matrix.pkl')
    joblib.dump(dict(game_popularity), f'{model_path}game_popularity.pkl')


def fit(data, model_path='models/saved_model/', n_components='auto', random_state=None):
    data = preprocess_data(data)

    # Инициализация статистик популярности игр
    game_popularity = defaultdict(int)
    for appid in data['appid']:
        game_popularity[int(appid)] += 1

    # Создаем user-item матрицу
    user_item_matrix = data.pivot_table(
        index='steamid',
        columns='appid',
        values='playtime_log',
        fill_value=0
    )

    # Автоматический подбор количества компонент
    n_features = user_item_matrix.shape[1]
    if n_components == 'auto':
        n_components = min(n_features - 1, 20)
    else:
        n_components = min(int(n_components), n_features - 1)

    print(f"Using {n_components} components for SVD (n_features={n_features})")

    # Создаем маппинги
    user_mapping = {user: idx for idx, user in enumerate(user_item_matrix.index)}
    game_mapping = {game: idx for idx, game in enumerate(user_item_matrix.columns)}

    # Дополняем статистики популярности
    for appid in user_item_matrix.columns:
        game_popularity[int(appid)] += (user_item_matrix[appid] > 0).sum()

    # Обучаем SVD
    sparse_matrix = csr_matrix(user_item_matrix.values)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_embeddings = svd.fit_transform(sparse_matrix)
    item_embeddings = svd.components_

    # Сохраняем модель
    save_model(
        model_path=model_path,
        svd=svd,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_mapping=user_mapping,
        game_mapping=game_mapping,
        user_item_matrix=user_item_matrix,
        game_popularity=game_popularity
    )

    print(f"Model successfully trained and saved to {model_path}")


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Train recommendation model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_path', type=str, default='models/saved_model/', help='Directory to save trained model')
    parser.add_argument('--n_components', type=str, default='auto', help='Number of SVD components (int or "auto")')

    args = parser.parse_args()

    # Загрузка данных
    data = pd.read_csv(args.data_path)

    # Обучение модели
    fit(data=data, model_path=args.model_path, n_components=args.n_components)


if __name__ == "__main__":
    main()
