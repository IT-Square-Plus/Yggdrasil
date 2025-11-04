"""
Natural language time expression parser for Yggdrasil.
Based on mcp-memory-service's time parser with Yggdrasil adaptations.
"""
import re
import logging
from datetime import datetime, timedelta, date, time
from typing import Tuple, Optional

logger = logging.getLogger("yggdrasil.utils.time_parser")

# Time of day mappings (24-hour format)
TIME_OF_DAY = {
    "morning": (5, 11),    # 5:00 AM - 11:59 AM
    "noon": (12, 12),      # 12:00 PM
    "afternoon": (13, 17), # 1:00 PM - 5:59 PM
    "evening": (18, 21),   # 6:00 PM - 9:59 PM
    "night": (22, 4),      # 10:00 PM - 4:59 AM (wraps around midnight)
    "midnight": (0, 0),    # 12:00 AM
}

# Regular expressions for various time patterns
PATTERNS = {
    "relative_days": re.compile(r'(?:(\d+)\s+days?\s+ago)|(?:yesterday)|(?:today)'),
    "relative_weeks": re.compile(r'(\d+)\s+weeks?\s+ago'),
    "relative_months": re.compile(r'(\d+)\s+months?\s+ago'),
    "relative_years": re.compile(r'(\d+)\s+years?\s+ago'),
    "last_period": re.compile(r'last\s+(day|week|month|year)'),
    "this_period": re.compile(r'this\s+(day|week|month|year)'),
    "month_name": re.compile(r'(january|february|march|april|may|june|july|august|september|october|november|december)'),
    "time_of_day": re.compile(r'(morning|afternoon|evening|night|noon|midnight)'),
    "recent": re.compile(r'recent|lately|recently'),
}

def parse_time_expression(query: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a natural language time expression and return timestamp range.

    Args:
        query: A natural language query with time expressions

    Returns:
        Tuple of (start_timestamp, end_timestamp), either may be None
    """
    query = query.lower().strip()

    try:
        # Relative days: "X days ago", "yesterday", "today"
        days_ago_match = PATTERNS["relative_days"].search(query)
        if days_ago_match:
            if "yesterday" in query:
                days = 1
            elif "today" in query:
                days = 0
            else:
                days = int(days_ago_match.group(1))

            target_date = date.today() - timedelta(days=days)

            # Check for time of day modifiers
            time_of_day_match = PATTERNS["time_of_day"].search(query)
            if time_of_day_match:
                # Narrow the range based on time of day
                return get_time_of_day_range(target_date, time_of_day_match.group(1))
            else:
                # Return the full day
                start_dt = datetime.combine(target_date, time.min)
                end_dt = datetime.combine(target_date, time.max)
                return start_dt.timestamp(), end_dt.timestamp()

        # Relative weeks: "X weeks ago"
        weeks_ago_match = PATTERNS["relative_weeks"].search(query)
        if weeks_ago_match:
            weeks = int(weeks_ago_match.group(1))
            target_date = date.today() - timedelta(weeks=weeks)
            # Get the start of the week (Monday)
            start_date = target_date - timedelta(days=target_date.weekday())
            end_date = start_date + timedelta(days=6)
            start_dt = datetime.combine(start_date, time.min)
            end_dt = datetime.combine(end_date, time.max)
            return start_dt.timestamp(), end_dt.timestamp()

        # Relative months: "X months ago"
        months_ago_match = PATTERNS["relative_months"].search(query)
        if months_ago_match:
            months = int(months_ago_match.group(1))
            current = datetime.now()
            # Calculate target month
            year = current.year
            month = current.month - months

            # Adjust year if month goes negative
            while month <= 0:
                year -= 1
                month += 12

            # Get first and last day of the month
            first_day = date(year, month, 1)
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)

            start_dt = datetime.combine(first_day, time.min)
            end_dt = datetime.combine(last_day, time.max)
            return start_dt.timestamp(), end_dt.timestamp()

        # Relative years: "X years ago"
        years_ago_match = PATTERNS["relative_years"].search(query)
        if years_ago_match:
            years = int(years_ago_match.group(1))
            current_year = datetime.now().year
            target_year = current_year - years
            start_dt = datetime(target_year, 1, 1, 0, 0, 0)
            end_dt = datetime(target_year, 12, 31, 23, 59, 59)
            return start_dt.timestamp(), end_dt.timestamp()

        # "Last X" expressions
        last_period_match = PATTERNS["last_period"].search(query)
        if last_period_match:
            period = last_period_match.group(1)
            return get_last_period_range(period)

        # "This X" expressions
        this_period_match = PATTERNS["this_period"].search(query)
        if this_period_match:
            period = this_period_match.group(1)
            return get_this_period_range(period)

        # Month names
        month_match = PATTERNS["month_name"].search(query)
        if month_match:
            month_name = month_match.group(1)
            return get_month_range(month_name)

        # Recent/fuzzy time expressions
        recent_match = PATTERNS["recent"].search(query)
        if recent_match:
            # Default to last 7 days for "recent"
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)
            return start_dt.timestamp(), end_dt.timestamp()

        # If no time expression is found, return None for both timestamps
        return None, None

    except Exception as e:
        logger.error(f"Error parsing time expression: {e}")
        return None, None

def get_time_of_day_range(target_date: date, time_period: str) -> Tuple[float, float]:
    """Get timestamp range for a specific time of day on a given date."""
    if time_period in TIME_OF_DAY:
        start_hour, end_hour = TIME_OF_DAY[time_period]

        # Handle periods that wrap around midnight
        if start_hour > end_hour:  # e.g., "night" = (22, 4)
            start_dt = datetime.combine(target_date, time(start_hour, 0))
            end_dt = datetime.combine(target_date + timedelta(days=1), time(end_hour, 59, 59))
        else:
            # Normal periods within a single day
            start_dt = datetime.combine(target_date, time(start_hour, 0))
            if end_hour == start_hour:  # For noon, midnight (specific hour)
                end_dt = datetime.combine(target_date, time(end_hour, 59, 59))
            else:
                end_dt = datetime.combine(target_date, time(end_hour, 59, 59))

        return start_dt.timestamp(), end_dt.timestamp()
    else:
        # Fallback to full day
        start_dt = datetime.combine(target_date, time.min)
        end_dt = datetime.combine(target_date, time.max)
        return start_dt.timestamp(), end_dt.timestamp()

def get_last_period_range(period: str) -> Tuple[float, float]:
    """Get timestamp range for 'last X' expressions."""
    now = datetime.now()
    today = date.today()

    if period == "day":
        # Last day = yesterday
        yesterday = today - timedelta(days=1)
        start_dt = datetime.combine(yesterday, time.min)
        end_dt = datetime.combine(yesterday, time.max)
    elif period == "week":
        # Last week = previous calendar week (Mon-Sun)
        last_monday = today - timedelta(days=today.weekday() + 7)
        last_sunday = last_monday + timedelta(days=6)
        start_dt = datetime.combine(last_monday, time.min)
        end_dt = datetime.combine(last_sunday, time.max)
    elif period == "month":
        # Last month = previous calendar month
        first_of_this_month = date(today.year, today.month, 1)
        if today.month == 1:
            last_month = 12
            last_month_year = today.year - 1
        else:
            last_month = today.month - 1
            last_month_year = today.year

        first_of_last_month = date(last_month_year, last_month, 1)
        last_of_last_month = first_of_this_month - timedelta(days=1)

        start_dt = datetime.combine(first_of_last_month, time.min)
        end_dt = datetime.combine(last_of_last_month, time.max)
    elif period == "year":
        # Last year = previous calendar year
        last_year = today.year - 1
        start_dt = datetime(last_year, 1, 1, 0, 0, 0)
        end_dt = datetime(last_year, 12, 31, 23, 59, 59)
    else:
        # Fallback - last 24 hours
        end_dt = now
        start_dt = end_dt - timedelta(days=1)

    return start_dt.timestamp(), end_dt.timestamp()

def get_this_period_range(period: str) -> Tuple[float, float]:
    """Get timestamp range for 'this X' expressions."""
    now = datetime.now()
    today = date.today()

    if period == "day":
        # This day = today
        start_dt = datetime.combine(today, time.min)
        end_dt = datetime.combine(today, time.max)
    elif period == "week":
        # This week = current calendar week (Mon-Sun)
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        start_dt = datetime.combine(monday, time.min)
        end_dt = datetime.combine(sunday, time.max)
    elif period == "month":
        # This month = current calendar month
        first_of_month = date(today.year, today.month, 1)
        if today.month == 12:
            first_of_next_month = date(today.year + 1, 1, 1)
        else:
            first_of_next_month = date(today.year, today.month + 1, 1)

        last_of_month = first_of_next_month - timedelta(days=1)

        start_dt = datetime.combine(first_of_month, time.min)
        end_dt = datetime.combine(last_of_month, time.max)
    elif period == "year":
        # This year = current calendar year
        start_dt = datetime(today.year, 1, 1, 0, 0, 0)
        end_dt = datetime(today.year, 12, 31, 23, 59, 59)
    else:
        # Fallback - current 24 hours
        end_dt = now
        start_dt = datetime.combine(today, time.min)

    return start_dt.timestamp(), end_dt.timestamp()

def get_month_range(month_name: str) -> Tuple[float, float]:
    """Get timestamp range for a named month."""
    # Map month name to number
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }

    if month_name in month_map:
        month_num = month_map[month_name]
        current_year = datetime.now().year

        # If the month is in the future for this year, use last year
        current_month = datetime.now().month
        year = current_year if month_num <= current_month else current_year - 1

        # Get first and last day of the month
        first_day = date(year, month_num, 1)
        if month_num == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month_num + 1, 1) - timedelta(days=1)

        start_dt = datetime.combine(first_day, time.min)
        end_dt = datetime.combine(last_day, time.max)
        return start_dt.timestamp(), end_dt.timestamp()
    else:
        return None, None

def extract_time_expression(query: str) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
    """
    Extract time-related expressions from a query and return the timestamps.

    Args:
        query: A natural language query that may contain time expressions

    Returns:
        Tuple of (cleaned_query, (start_timestamp, end_timestamp))
        The cleaned_query has time expressions removed
    """
    # Check for time expressions
    time_expressions = [
        r'\b\d+\s+days?\s+ago\b',
        r'\byesterday\b',
        r'\btoday\b',
        r'\b\d+\s+weeks?\s+ago\b',
        r'\b\d+\s+months?\s+ago\b',
        r'\b\d+\s+years?\s+ago\b',
        r'\blast\s+(day|week|month|year)\b',
        r'\bthis\s+(day|week|month|year)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\bin\s+the\s+(morning|afternoon|evening|night|noon|midnight)\b',
        r'\brecent|lately|recently\b',
    ]

    # Combine all patterns
    combined_pattern = '|'.join(f'({expr})' for expr in time_expressions)
    combined_regex = re.compile(combined_pattern, re.IGNORECASE)

    # Find all matches
    matches = list(combined_regex.finditer(query))
    if not matches:
        return query, (None, None)

    # Extract the time expressions
    time_expressions_found = []
    for match in matches:
        span = match.span()
        expression = query[span[0]:span[1]]
        time_expressions_found.append(expression)

    # Parse time expressions to get timestamps
    full_time_expression = ' '.join(time_expressions_found)
    start_ts, end_ts = parse_time_expression(full_time_expression)

    # Remove time expressions from the query
    cleaned_query = query
    for expr in time_expressions_found:
        cleaned_query = cleaned_query.replace(expr, '')

    # Clean up multiple spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return cleaned_query, (start_ts, end_ts)
