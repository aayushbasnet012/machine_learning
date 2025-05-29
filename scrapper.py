import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import re


def scrape_injury_data():
    """
    Scrapes football injury data from football-lineups.com and formats it as specified.
    Returns a dictionary with player names as keys and their injury information as values.
    """
    # URL to scrape
    url = "https://www.football-lineups.com/injuries/"

    # Send request with headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return {}

    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Initialize the dictionary to store all players' data
    all_players_data = {}

    # Find the tables that contain injury data
    # The site typically has multiple tables, one for each team
    injury_tables = soup.find_all('table', class_='injtable')

    # Process each table (team)
    for table in injury_tables:
        # Get team name (it's usually in a previous sibling element)
        team_name = "Unknown Team"
        team_header = table.find_previous('div', class_='injteam')
        if team_header:
            team_name = team_header.text.strip()

        # Process each row in the table
        rows = table.find_all('tr')
        # Skip header row
        for row in rows[1:]:
            columns = row.find_all('td')

            # Ensure we have enough columns
            if len(columns) >= 4:
                # Extract player name
                player_name = columns[0].text.strip()

                # If player already exists in our data, append to their record
                if player_name not in all_players_data:
                    all_players_data[player_name] = {
                        "Team": team_name,
                        "Id": float('nan'),  # Using NaN as specified
                        "Injury_dates": [],
                        "Injury_duration": []
                    }

                # Extract injury date
                injury_date_text = columns[1].text.strip()
                try:
                    # Try to parse the date (format may vary)
                    injury_date = parse_date(injury_date_text)

                    # Extract injury duration
                    duration_text = columns[3].text.strip()
                    duration = extract_number(duration_text)

                    if injury_date and duration is not None:
                        all_players_data[player_name]["Injury_dates"].append(
                            injury_date)
                        all_players_data[player_name]["Injury_duration"].append(
                            duration)
                except Exception as e:
                    print(f"Error processing data for {player_name}: {e}")

    return all_players_data


def parse_date(date_text):
    """
    Parse date string into YYYY-MM-DD format.
    Handles multiple potential input formats.
    """
    date_text = date_text.strip()

    # Try different date formats that might be encountered
    date_formats = [
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%d.%m.%Y',
        '%d %b %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%B %d, %Y'
    ]

    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_text, fmt)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # If all formats fail, try to extract date parts using regex
    # This is a fallback for unusual formats
    date_pattern = r'(\d{1,2})[-./\s](\d{1,2}|[A-Za-z]+)[-./\s](\d{2,4})'
    match = re.search(date_pattern, date_text)

    if match:
        day, month, year = match.groups()

        # Convert month name to number if needed
        if month.isalpha():
            month_names = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = month_names.get(month.lower()[:3], 1)

        # Ensure year is 4 digits
        if len(str(year)) == 2:
            year = '20' + str(year) if int(year) < 50 else '19' + str(year)

        try:
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            pass

    # If all attempts fail, return original string
    print(f"Warning: Could not parse date '{date_text}'")
    return date_text


def extract_number(text):
    """
    Extract the duration number from text like "22 days" or "3 weeks"
    Converts weeks to days if necessary
    """
    if not text:
        return None

    # Look for number followed by time unit
    number_match = re.search(
        r'(\d+)\s*(day|days|week|weeks|month|months)?', text)

    if number_match:
        number = int(number_match.group(1))
        unit = number_match.group(2) if number_match.group(2) else 'days'

        # Convert to days if needed
        if 'week' in unit:
            return number * 7
        elif 'month' in unit:
            return number * 30
        else:
            return number

    # Try to find just a number
    number_only = re.search(r'(\d+)', text)
    if number_only:
        return int(number_only.group(1))

    return None


def save_to_json(data, filename="player_injuries.json"):
    """
    Save the data to a JSON file with pretty formatting
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


def main():
    print("Scraping football injury data...")
    injury_data = scrape_injury_data()

    # Save data to JSON file
    save_to_json(injury_data)

    # Print a sample of the data
    print("\nSample of the scraped data:")
    sample_count = min(3, len(injury_data))
    sample_players = list(injury_data.keys())[:sample_count]

    for player in sample_players:
        print(f"\n{player}:", json.dumps(injury_data[player], indent=8))

    print(f"\nTotal players scraped: {len(injury_data)}")


main()
