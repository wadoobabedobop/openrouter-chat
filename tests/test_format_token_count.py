import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

from streamlit_app import format_token_count

@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0"),
        (5, "5"),
        (999, "999"),
        (None, "N/A"),
    ],
)
def test_format_token_count_below_thousand(value, expected):
    assert format_token_count(value) == expected

@pytest.mark.parametrize(
    "value,expected",
    [
        (1000, "1k"),
        (1500, "1.5k"),
        (12000, "12k"),
        (999999, "1000k"),
    ],
)
def test_format_token_count_thousands(value, expected):
    assert format_token_count(value) == expected

@pytest.mark.parametrize(
    "value,expected",
    [
        (1000000, "1M"),
        (1500000, "1.5M"),
        (5000000, "5M"),
    ],
)
def test_format_token_count_millions(value, expected):
    assert format_token_count(value) == expected
