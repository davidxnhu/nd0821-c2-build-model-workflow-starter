import pandas as pd
import numpy as np
import scipy.stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    logger.info("Test column names: column names expected: %s",
                 list(expected_colums))
    logger.info("Test column names: column names in dataset: %s",
                 list(these_columns))

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    logger.info("Test column names: neighbourhood group expected: %s",
                 set(known_names))
    logger.info("Test column names: neighbourhood group in dataset: %s",
                 set(neigh))

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    logger.info("Test proper boundaries: unexpected items are %s", np.sum(~idx))

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    logger.info(
        "Test if the distribution of the new data is significantly different than that of the reference dataset")
    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    """
    Test row count is into a good range
    """

    logger.info("Test row count: items are %s", data.shape[0])
    assert 15000< data.shape[0] <1000000


def test_price_range(data, min_price, max_price):
    """
    Check price range is between min and max prices
    """
    in_between_price_row = data['price'].between(min_price, max_price).shape[0]
    logging.info("Price range test, items in range are: %s", in_between_price_row)
    assert in_between_price_row == data.shape[0]
