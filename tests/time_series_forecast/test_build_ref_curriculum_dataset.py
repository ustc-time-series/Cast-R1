from recipe.time_series_forecast.build_ref_curriculum_dataset import (
    assign_quantile_bands,
    compute_reference_value,
    compute_normalized_reference_value,
)


def test_assign_quantile_bands_orders_low_mid_high():
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    bands = assign_quantile_bands(values)

    assert bands == [1, 1, 2, 2, 3, 3]


def test_assign_quantile_bands_handles_equal_values_with_all_bands():
    values = [1.0, 1.0, 1.0]

    bands = assign_quantile_bands(values)

    assert sorted(set(bands)) == [1, 2, 3]


def test_compute_reference_value_prioritizes_error_and_variation():
    flat_history = [1.0, 1.0, 1.0]
    easy_future = [1.0, 1.0, 1.0]
    hard_future = [2.0, 4.0, 8.0]

    easy = compute_reference_value(flat_history, easy_future)
    hard = compute_reference_value(flat_history, hard_future)

    assert hard > easy


def test_compute_normalized_reference_value_is_scale_robust():
    history = [10.0, 11.0, 12.0, 13.0]
    future = [14.0, 15.0, 16.0, 17.0]
    scaled_history = [value * 10.0 for value in history]
    scaled_future = [value * 10.0 for value in future]

    base = compute_normalized_reference_value(history, future)
    scaled = compute_normalized_reference_value(scaled_history, scaled_future)

    assert abs(base - scaled) < 0.25


def test_compute_normalized_reference_value_prioritizes_relative_error():
    history = [10.0, 10.0, 10.0, 10.0]
    easy_future = [10.0, 10.2, 9.8, 10.1]
    hard_future = [14.0, 7.0, 16.0, 4.0]

    easy = compute_normalized_reference_value(history, easy_future)
    hard = compute_normalized_reference_value(history, hard_future)

    assert hard > easy
