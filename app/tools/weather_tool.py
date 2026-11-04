import calendar
from datetime import datetime, timedelta
import re
import time
from typing import Any, Dict, Optional, Tuple
import httpx
from langchain_core.tools import tool

from app.config import settings
from app.utils.logger import TraceLogger, logger

# City coordinates mapping
CITY_COORDINATES: Dict[str, Dict[str, Any]] = {
    "tokyo": {
        "city": "Tokyo",
        "country": "Japan",
        "latitude": 35.6762,
        "longitude": 139.6503,
        "timezone": "Asia/Tokyo",
    },
    "bangkok": {
        "city": "Bangkok",
        "country": "Thailand",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "timezone": "Asia/Bangkok",
    },
    "barcelona": {
        "city": "Barcelona",
        "country": "Spain",
        "latitude": 41.3851,
        "longitude": 2.1734,
        "timezone": "Europe/Madrid",
    },
    "reykjavik": {
        "city": "Reykjavik",
        "country": "Iceland",
        "latitude": 64.1466,
        "longitude": -21.9426,
        "timezone": "Atlantic/Reykjavik",
    },
}

# WMO Weather interpretation codes
WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Monthly historical climate averages (temp_range_c [min, max], conditions, rain_risk)
HISTORICAL_CLIMATE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "tokyo": {
        "january": {"temp_range_c": [2, 10], "conditions": "Cold and sunny, dry", "rainfall_mm": 50},
        "february": {"temp_range_c": [3, 11], "conditions": "Cold, crisp days, occasional snow", "rainfall_mm": 55},
        "march": {"temp_range_c": [6, 14], "conditions": "Mild, early cherry blossoms", "rainfall_mm": 115},
        "april": {"temp_range_c": [11, 19], "conditions": "Pleasant and mild, spring weather", "rainfall_mm": 130},
        "may": {"temp_range_c": [15, 23], "conditions": "Warm and comfortable", "rainfall_mm": 140},
        "june": {"temp_range_c": [19, 26], "conditions": "Warm and humid, rainy season (Tsuyu)", "rainfall_mm": 170},
        "july": {"temp_range_c": [23, 30], "conditions": "Hot, humid, sunny spells", "rainfall_mm": 155},
        "august": {"temp_range_c": [24, 31], "conditions": "Very hot, high humidity, typhoon risk", "rainfall_mm": 170},
        "september": {"temp_range_c": [20, 27], "conditions": "Warm, humid, occasional typhoons", "rainfall_mm": 210},
        "october": {"temp_range_c": [14, 22], "conditions": "Pleasant, mild autumn weather", "rainfall_mm": 200},
        "november": {"temp_range_c": [9, 17], "conditions": "Cool, clear skies, autumn foliage", "rainfall_mm": 95},
        "december": {"temp_range_c": [4, 12], "conditions": "Cold, dry and sunny", "rainfall_mm": 50},
    },
    "bangkok": {
        "january": {"temp_range_c": [22, 33], "conditions": "Warm and dry, very pleasant", "rainfall_mm": 15},
        "february": {"temp_range_c": [24, 34], "conditions": "Warm to hot, low rain", "rainfall_mm": 25},
        "march": {"temp_range_c": [26, 35], "conditions": "Hot and increasingly humid", "rainfall_mm": 40},
        "april": {"temp_range_c": [27, 36], "conditions": "Very hot and sunny (hottest month)", "rainfall_mm": 80},
        "may": {"temp_range_c": [26, 34], "conditions": "Hot with frequent monsoon showers", "rainfall_mm": 220},
        "june": {"temp_range_c": [26, 33], "conditions": "Hot and humid, afternoon showers", "rainfall_mm": 180},
        "july": {"temp_range_c": [25, 33], "conditions": "Humid with regular rainfall bursts", "rainfall_mm": 175},
        "august": {"temp_range_c": [25, 33], "conditions": "Cloudy, humid with frequent rain", "rainfall_mm": 220},
        "september": {"temp_range_c": [25, 32], "conditions": "Peak monsoon season, heavy rain showers", "rainfall_mm": 330},
        "october": {"temp_range_c": [25, 32], "conditions": "Monsoon easing, heavy rain early month", "rainfall_mm": 240},
        "november": {"temp_range_c": [23, 32], "conditions": "Start of cool/dry season, comfortable", "rainfall_mm": 50},
        "december": {"temp_range_c": [21, 31], "conditions": "Cooler evenings, lowest humidity, dry", "rainfall_mm": 10},
    },
    "barcelona": {
        "january": {"temp_range_c": [5, 14], "conditions": "Crisp and mild, sunny intervals", "rainfall_mm": 40},
        "february": {"temp_range_c": [6, 15], "conditions": "Mild, occasional rain", "rainfall_mm": 35},
        "march": {"temp_range_c": [8, 17], "conditions": "Mild spring, pleasant afternoons", "rainfall_mm": 45},
        "april": {"temp_range_c": [11, 19], "conditions": "Warm and pleasant, occasional spring shower", "rainfall_mm": 50},
        "may": {"temp_range_c": [14, 23], "conditions": "Warm and sunny, beach weather begins", "rainfall_mm": 55},
        "june": {"temp_range_c": [18, 26], "conditions": "Warm and sunny, great beach weather", "rainfall_mm": 30},
        "july": {"temp_range_c": [21, 29], "conditions": "Hot, humid, clear blue skies", "rainfall_mm": 20},
        "august": {"temp_range_c": [21, 29], "conditions": "Hot and sunny, peak tourist season", "rainfall_mm": 60},
        "september": {"temp_range_c": [18, 26], "conditions": "Warm with pleasant sea temperatures, occasional storms", "rainfall_mm": 85},
        "october": {"temp_range_c": [13, 22], "conditions": "Mild autumn, highest annual rainfall", "rainfall_mm": 90},
        "november": {"temp_range_c": [9, 17], "conditions": "Cool and pleasant, decreasing crowds", "rainfall_mm": 60},
        "december": {"temp_range_c": [6, 15], "conditions": "Cool and brisk, mild Mediterranean winter", "rainfall_mm": 45},
    },
    "reykjavik": {
        "january": {"temp_range_c": [-3, 2], "conditions": "Cold, snow/ice, Northern Lights, 4-5h daylight", "rainfall_mm": 85},
        "february": {"temp_range_c": [-2, 3], "conditions": "Cold, windy, frequent snow, Northern Lights", "rainfall_mm": 85},
        "march": {"temp_range_c": [-2, 3], "conditions": "Cold, variable snow/rain, increasing light", "rainfall_mm": 80},
        "april": {"temp_range_c": [1, 6], "conditions": "Chilly, unpredictable spring weather", "rainfall_mm": 60},
        "may": {"temp_range_c": [4, 10], "conditions": "Cool, long days, occasional light rain", "rainfall_mm": 50},
        "june": {"temp_range_c": [7, 13], "conditions": "Mild, near 24-hour daylight (Midnight Sun)", "rainfall_mm": 45},
        "july": {"temp_range_c": [9, 15], "conditions": "Mildest month, comfortable outdoors, daylight", "rainfall_mm": 50},
        "august": {"temp_range_c": [8, 14], "conditions": "Mild, occasional rain and wind", "rainfall_mm": 65},
        "september": {"temp_range_c": [5, 11], "conditions": "Cool, autumn colours, Northern Lights return", "rainfall_mm": 75},
        "october": {"temp_range_c": [2, 7], "conditions": "Chilly, windy, mixed rain/sleet", "rainfall_mm": 85},
        "november": {"temp_range_c": [-1, 4], "conditions": "Cold, frosty, short daylight, Northern Lights", "rainfall_mm": 80},
        "december": {"temp_range_c": [-3, 2], "conditions": "Cold, snowy, darkest month (4h daylight)", "rainfall_mm": 95},
    },
}

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]


def resolve_city_info(city_name: str) -> Optional[Dict[str, Any]]:
    """Lookup city coordinates and configuration."""
    if not city_name:
        return None
    city_clean = city_name.strip().lower()
    for key, info in CITY_COORDINATES.items():
        if key in city_clean or info["city"].lower() in city_clean:
            return info
    return None


def parse_target_date_or_month(date_input: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Parse user input into either a specific datetime or a month string."""
    if not date_input or not date_input.strip():
        return datetime.now(), None

    d_clean = date_input.strip().lower()

    if d_clean in ("today", "current", "now"):
        return datetime.now(), None
    if d_clean == "tomorrow":
        return datetime.now() + timedelta(days=1), None

    # Check for month names
    for m in MONTH_NAMES:
        if m in d_clean or m[:3] in d_clean:
            return None, m

    # Try standard date formats
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%B %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(d_clean, fmt), None
        except ValueError:
            continue

    # Fallback to current date
    return datetime.now(), None


@tool
def get_weather_forecast(city: str, date_or_month: str = "") -> dict:
    """
    Get the weather forecast for a given destination.
    """
    t0 = time.time()
    TraceLogger.tool_call("get_weather_forecast", {"city": city, "date_or_month": date_or_month})

    # Validate city
    city_info = resolve_city_info(city)
    if not city_info:
        duration_ms = (time.time() - t0) * 1000
        supported = ", ".join([v["city"] for v in CITY_COORDINATES.values()])
        error_res = {
            "status": "error",
            "city": city,
            "error": f"Unsupported destination: '{city}'. TripMate weather currently supports: {supported}.",
            "data_type": "none",
        }
        TraceLogger.tool_status("get_weather_forecast", "error", duration_ms, f"Unsupported city: {city}")
        return error_res

    city_key = city_info["city"].lower()
    city_name = city_info["city"]
    target_date, target_month = parse_target_date_or_month(date_or_month)

    # If user provided a month name or no specific date, use seasonal climate estimate
    if target_month:
        month_data = HISTORICAL_CLIMATE.get(city_key, {}).get(target_month)
        if month_data:
            duration_ms = (time.time() - t0) * 1000
            res = {
                "status": "success",
                "city": city_name,
                "country": city_info["country"],
                "period": target_month.capitalize(),
                "data_type": "seasonal_estimate",
                "source": "Open-Meteo Historical Climate / Meteorological Averages",
                "temp_range_c": month_data["temp_range_c"],
                "conditions": month_data["conditions"],
                "rainfall_mm": month_data["rainfall_mm"],
                "disclaimer": "This is a seasonal climate estimate based on historical averages, not a live real-time forecast.",
            }
            TraceLogger.tool_status(
                "get_weather_forecast",
                "success",
                duration_ms,
                f"{city_name} seasonal estimate for {target_month.capitalize()}: {month_data['conditions']}",
            )
            return res

    # If specific date, check if it falls within live forecast horizon (today to today + 14 days)
    now = datetime.now()
    if target_date:
        days_diff = (target_date.date() - now.date()).days
        if 0 <= days_diff <= 14:
            # Within live forecast horizon: Call Open-Meteo
            try:
                base_url = settings.weather_api_base_url
                url = (
                    f"{base_url}/v1/forecast"
                    f"?latitude={city_info['latitude']}&longitude={city_info['longitude']}"
                    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
                    f"&timezone={city_info['timezone']}"
                )

                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                daily = data.get("daily", {})
                dates = daily.get("time", [])
                target_str = target_date.strftime("%Y-%m-%d")

                idx = 0
                if target_str in dates:
                    idx = dates.index(target_str)

                t_max = daily.get("temperature_2m_max", [None])[idx]
                t_min = daily.get("temperature_2m_min", [None])[idx]
                precip = daily.get("precipitation_sum", [0.0])[idx]
                w_code = daily.get("weather_code", [0])[idx]
                condition = WMO_CODE_MAP.get(w_code, "Partly cloudy")

                duration_ms = (time.time() - t0) * 1000
                res = {
                    "status": "success",
                    "city": city_name,
                    "country": city_info["country"],
                    "date": dates[idx] if idx < len(dates) else target_str,
                    "data_type": "live_forecast",
                    "source": "Open-Meteo API",
                    "temp_range_c": [t_min, t_max],
                    "conditions": condition,
                    "precipitation_mm": precip,
                    "disclaimer": "Live forecast provided by Open-Meteo.",
                }
                TraceLogger.tool_status(
                    "get_weather_forecast",
                    "success",
                    duration_ms,
                    f"{city_name} live forecast ({dates[idx]}): {condition}, {t_min}°C to {t_max}°C",
                )
                return res

            except httpx.TimeoutException:
                logger.warning(f"Open-Meteo API timed out for {city_name}. Falling back to seasonal estimate.")
            except Exception as e:
                logger.warning(f"Open-Meteo API error for {city_name}: {e}. Falling back to seasonal estimate.")

    # Dates beyond live horizon (or fallback if API call fails)
    m_name = (target_date or now).strftime("%B").lower()
    month_data = HISTORICAL_CLIMATE.get(city_key, {}).get(m_name, {
        "temp_range_c": [10, 20],
        "conditions": "Moderate seasonal weather",
        "rainfall_mm": 50,
    })

    duration_ms = (time.time() - t0) * 1000
    res = {
        "status": "success",
        "city": city_name,
        "country": city_info["country"],
        "date_or_period": target_date.strftime("%Y-%m-%d") if target_date else m_name.capitalize(),
        "data_type": "seasonal_estimate",
        "source": "Open-Meteo Historical Climate / Meteorological Averages",
        "temp_range_c": month_data["temp_range_c"],
        "conditions": month_data["conditions"],
        "rainfall_mm": month_data["rainfall_mm"],
        "disclaimer": (
            "This is a seasonal climate estimate based on historical averages because the requested "
            "date is beyond the 14-day live forecast window or a general month was requested."
        ),
    }
    TraceLogger.tool_status(
        "get_weather_forecast",
        "success",
        duration_ms,
        f"{city_name} seasonal estimate for {m_name.capitalize()}: {month_data['conditions']}",
    )
    return res

