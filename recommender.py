import numpy as np
import joblib
from collections import defaultdict


class CollaborativeRecommender:
    def __init__(self, model_path='models/saved_model/'):
        self.model_path = model_path
        self.is_trained = False
        self.load_model()

    def load_model(self):
        try:
            self.svd = joblib.load(f'{self.model_path}svd_model.pkl')
            self.user_embeddings = joblib.load(f'{self.model_path}user_embeddings.pkl')
            self.item_embeddings = joblib.load(f'{self.model_path}item_embeddings.pkl')
            self.user_mapping = joblib.load(f'{self.model_path}user_mapping.pkl')
            self.game_mapping = joblib.load(f'{self.model_path}game_mapping.pkl')
            self.game_mapping_inv = {v: k for k, v in self.game_mapping.items()}
            self.user_item_matrix = joblib.load(f'{self.model_path}user_item_matrix.pkl')
            self.game_popularity = defaultdict(int, joblib.load(f'{self.model_path}game_popularity.pkl'))
            self.is_trained = True
        except FileNotFoundError:
            self.svd = None
            self.user_embeddings = None
            self.item_embeddings = None
            self.user_mapping = None
            self.game_mapping = None
            self.game_mapping_inv = None
            self.user_item_matrix = None
            self.game_popularity = None
            self.is_trained = False

    def recommend(self, user_id=None, user_games=None, n=5): # diversity=0.2
        '''
        user_id - int,
        user_games - list
        '''
        try:
            # Получаем сыгранные игры с гарантированной конвертацией в int
            played_games = self.get_played_games(user_id, user_games)
            print('Played games:', *played_games, sep=' ')

            recommendations = []
            # 1. Персональные рекомендации для известного пользователя
            if user_id is not None and user_id in self.user_mapping:
                user_idx = self.user_mapping[user_id]
                personal_scores = {}
                for appid, idx in self.game_mapping.items():
                    appid_int = int(appid)
                    if appid_int not in played_games:
                        score = float(self.user_embeddings[user_idx] @ self.item_embeddings[:, idx])
                        personal_scores[appid_int] = score
                if personal_scores:
                    # это хорошая идея, но она может портить результаты теста. НЕ УДАЛЯТЬ
                    #recs = self.diversify_recommendations(personal_scores, n, diversity)
                    # Дополнительная проверка на исключение сыгранных игр
                    # return [game for game in recs if not filter_played or game not in played_games][:n]

                    # return personal_scores[:n]
                    recommendations.extend(sorted(personal_scores.items(), key=lambda x: -x[1])[:n])

            # 2. Гибридные рекомендации по списку игр
            if user_games and len(user_games) > 0:
                hybrid_scores = self.get_hybrid_scores(user_games, played_games)

                if hybrid_scores:
                    # НОВОЕ ДО RETURN
                    combined_scores = {}
                    # Добавляем базовые рекомендации с весом 1.0
                    for appid, score in recommendations:
                        combined_scores[appid] = score * 1.0  # полный вес старым предпочтениям
                    # Добавляем влияние новых игр с весом 0.5
                    for appid, score in hybrid_scores.items():
                        if appid in combined_scores:
                            combined_scores[appid] += score * 0.5  # Усиливаем имеющиеся
                        else:
                            combined_scores[appid] = score * 0.5  # Добавляем новые
                    recommendations = sorted(combined_scores.items(), key=lambda x: -x[1])
                    # recs = self.diversify_recommendations(hybrid_scores, n, diversity/2)
                    # return [game for game in recs if not filter_played or game not in played_games][:n]
                    # return hybrid_scores[:n] #last

            # 3. Холодный старт
            if not recommendations:
                cold_start_recs = self.get_cold_start_recommendations(n*2) #, exclude=played_games
                weights = np.arange(len(cold_start_recs), 0, -1) ** 0.5
                weights = weights / weights.sum()
                selected = np.random.choice(len(cold_start_recs), size=n, replace=False, p=weights)
                recommendations = [(cold_start_recs[i], 0) for i in selected]  # [cold_start_recs[i] for i in selected]

            # Возвращаем топ-N appid
            return [appid for appid, score in recommendations[:n]]
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            if self.is_trained:
                # Безопасный fallback
                return list(self.game_mapping.keys())[:n]
            else:
                print('Модель не обучена.')

    def get_played_games(self, user_id, user_games):
        played = set()

        # игры из матрицы для известных пользователей
        if user_id is not None and user_id in self.user_mapping:
            user_idx = self.user_mapping[user_id]
            played.update([
                int(appid)
                for appid in self.user_item_matrix.columns
                if self.user_item_matrix.iloc[user_idx][appid] > 0
            ])

        # игры из списка (с конвертацией в int)
        if user_games is not None:
            played.update(
                int(entry[0]) if isinstance(entry, tuple) else int(entry)
                for entry in user_games
            )

        return played

    # для разнообразия рекомендаций - чтобы не одни и те же попадались
    # можно попробовать без этого, закомментировать
    # нужно будет, скорее всего, обновить
    # def diversify_recommendations(self, scores, n, diversity_factor):
    #     if not scores or len(scores) < 3:  # Не добавляем разнообразие для малого числа вариантов
    #         return sorted(scores.keys(), key=lambda x: -scores[x])[:n]
    #
    #     items = sorted(scores.items(), key=lambda x: -x[1])
    #
    #     # Берем топ 3*n для лучшего разнообразия
    #     top_items = items[:3*n]
    #
    #     # Веса с экспоненциальным затуханием
    #     ranks = np.arange(len(top_items))
    #     weights = np.exp(-diversity_factor * ranks)
    #     weights = weights / weights.sum()
    #
    #     # Выбираем случайно, но с учетом весов
    #     selected = np.random.choice(
    #         len(top_items),
    #         size=min(n, len(top_items)),
    #         replace=False,
    #         p=weights
    #     )
    #
    #     # Сортируем выбранные по убыванию релевантности
    #     result = [top_items[i][0] for i in selected]
    #     result.sort(key=lambda x: -scores[x])
    #
    #     return result

    # по списку игр (интересная идея с вектором предпочтений и логарифмом для сглаживания разницы)
    def get_hybrid_scores(self, user_games, played_games):
        pref_vector = np.zeros(len(self.game_mapping))
        for entry in user_games:
            appid = int(entry[0] if isinstance(entry, tuple) else entry)
            playtime = entry[1] if isinstance(entry, tuple) else 1
            if appid in self.game_mapping:
                pref_vector[self.game_mapping[appid]] = np.log1p(playtime)

        if pref_vector.sum() > 0:
            pref_vector /= pref_vector.sum()
            scores = {
                int(appid): pref_vector @ self.item_embeddings.T @ self.item_embeddings[:, idx]
                for appid, idx in self.game_mapping.items()
                if int(appid) not in played_games
            }
            return scores
        return {}

    # на основе популярности - для новых
    # популярность надо как-то получить! может, даже при обучении?
    def get_cold_start_recommendations(self, n, exclude=None):
        exclude = exclude or set()

        # Получаем все игры из game_mapping, если popularity пуст
        if not self.game_popularity:
            games = [int(appid) for appid in self.game_mapping.keys()]
            return games[:n] if games else []

        # Создаем список популярных игр
        popular_games = [
            (appid, count)
            for appid, count in self.game_popularity.items()
            if int(appid) not in exclude and int(appid) in self.game_mapping
        ]

        # Если нет популярных, берем любые доступные
        if not popular_games:
            available = [int(appid) for appid in self.game_mapping.keys() if int(appid) not in exclude]
            return available[:n]

        popular_games.sort(key=lambda x: -x[1])
        return [int(appid) for appid, _ in popular_games[:n]]
