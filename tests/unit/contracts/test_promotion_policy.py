# tests/unit/contracts/test_promotion_policy.py
from dags.house_prices_train_model import should_promote_model


def test_should_promote_no_previous_model():
    assert should_promote_model(None, None, new_rmse=50000, new_r2=0.2) is True


def test_should_promote_better_rmse_and_r2():
    assert should_promote_model(
        old_rmse=60000,
        old_r2=0.18,
        new_rmse=55000,
        new_r2=0.20,
    ) is True


def test_should_not_promote_worse_rmse():
    assert should_promote_model(
        old_rmse=50000,
        old_r2=0.20,
        new_rmse=55000,  # peor (más alto)
        new_r2=0.22,     # mejor r2 pero no compensa
    ) is False


def test_should_not_promote_worse_r2():
    assert should_promote_model(
        old_rmse=50000,
        old_r2=0.20,
        new_rmse=48000,  # mejor
        new_r2=0.18,     # peor
    ) is False


def test_should_promote_equal_metrics():
    assert should_promote_model(
        old_rmse=50000,
        old_r2=0.20,
        new_rmse=50000,  # igual
        new_r2=0.20,     # igual
    ) is True
