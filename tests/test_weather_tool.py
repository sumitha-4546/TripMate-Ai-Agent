from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import httpx
import pytest

from app.tools.weather_tool import get_weather_forecast


def test_valid_destination_seasonal():
    result = get_weather_forecast.invoke({"city": "Tokyo", "date_or_month": "December"})

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["city"] == "Tokyo"
    assert result["country"] == "Japan"
    assert result["data_type"] == "seasonal_estimate"
    assert "temp_range_c" in result
    assert result["temp_range_c"] == [4, 12]
    assert "Cold" in result["conditions"] or "dry" in result["conditions"]
    assert "seasonal" in result["disclaimer"].lower()


def test_invalid_destination():
    result = get_weather_forecast.invoke({"city": "Atlantis", "date_or_month": "today"})

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert "Unsupported destination" in result["error"]
    assert "Tokyo" in result["error"]
    assert "Bangkok" in result["error"]


def test_missing_date_handled_gracefully():
    result = get_weather_forecast.invoke({"city": "Bangkok", "date_or_month": ""})

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["city"] == "Bangkok"
    assert "temp_range_c" in result
    assert "conditions" in result


def test_api_timeout_fallback():
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Mock httpx.Client to raise TimeoutException
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Open-Meteo connection timed out")):
        result = get_weather_forecast.invoke({"city": "Barcelona", "date_or_month": tomorrow})

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["city"] == "Barcelona"
    assert result["data_type"] == "seasonal_estimate"
    assert "historical averages" in result["disclaimer"].lower()


def test_api_failure_fallback():
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Server Error", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.Client.get", return_value=mock_resp):
        result = get_weather_forecast.invoke({"city": "Reykjavik", "date_or_month": tomorrow})

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["city"] == "Reykjavik"
    assert result["data_type"] == "seasonal_estimate"


def test_forecast_horizon_limitation():
    # 45 days in future
    future_date = (datetime.now() + timedelta(days=45)).strftime("%Y-%m-%d")

    result = get_weather_forecast.invoke({"city": "Reykjavik", "date_or_month": future_date})

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["city"] == "Reykjavik"
    assert result["data_type"] == "seasonal_estimate"
    assert result["data_type"] != "live_forecast"
    assert "beyond the 14-day live forecast window" in result["disclaimer"]

