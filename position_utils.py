# In your analysis scripts
from position_utils import get_player_position, categorize_position, get_position_statistics

# Get a player's position
position = get_player_position("Player Name")
print(f"Position: {position}")

# Get broad category
broad_cat = get_broad_position_category(position)
print(f"Category: {broad_cat}")

# Get all forwards
forwards = get_all_players_with_position(broad_category="Forward")

# Get position statistics
stats = get_position_statistics()
print(f"Total players: {stats['total_players']}")