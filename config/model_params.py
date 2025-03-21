from scipy.stats import uniform, randint

# Model parameters
LIGHTGBM_PARAMS = {
            'n_estimators': randint(100,500),
            'max_depth': randint(5, 50),
            'learning_rate': uniform(0.01, 0.2),
            'num_leaves': randint(20, 100),
            'boosting_type': ['gbdt', 'dart','goss']
}

RANDOM_SEACH_PARAMS = {
    'n_iter': 4,
    'scoring': 'accuracy',
    'cv': 2,
    'verbose': 2,
    'n_jobs': -1,
    'random_state': 42
}