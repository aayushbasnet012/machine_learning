import numpy as np
import json

def extract_player_positions_from_tensor():
    """Extract player positions from the processed tensor data"""
    
    # Load the saved tensor data
    all_players_tensor = np.load('positional_all_players_tensor.npy')
    
    # Load player info to get player names in order
    with open('all_player_info_new_dates.json', 'r', encoding='utf-8') as f:
        player_info = json.load(f)
    
    # Define categories (same order as in your processing code)
    categories = ['Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
                 'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
                 'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
                 'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
                 'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
                 'Right Midfield', 'Left Wing', 'Left Center Forward', 
                 'Center Forward', 'Right Wing', 'Right Center Forward',
                 "under_pressure", "50_50", "Pass", "Shot", "Dribble", "Pressure", "Duel", 
                 "Interception", "Foul Committed", "Carry",
                 "minutes_played", "injury_duration", "injury"]
    
    # Position categories only (first 24 items)
    position_categories = categories[:24]
    
    # Dictionary to store player positions
    player_positions = {}
    
    # Get player names in the same order as processed
    player_names = list(player_info.keys())
    
    # Extract position for each player
    for player_idx, player_name in enumerate(player_names):
        if player_idx < all_players_tensor.shape[0]:  # Make sure we don't exceed tensor bounds
            
            # Get all matches data for this player
            player_matches = all_players_tensor[player_idx]
            
            # Find the position by looking for which position category has a 1
            player_position = 'Unknown'
            
            # Check across all matches to find the position
            for match_idx in range(player_matches.shape[0]):
                match_data = player_matches[match_idx]
                
                # Check position categories (indices 0-23)
                for pos_idx, position in enumerate(position_categories):
                    if pos_idx < len(match_data) and match_data[pos_idx] == 1:
                        player_position = position
                        break
                
                # If we found a position, break out of match loop
                if player_position != 'Unknown':
                    break
            
            player_positions[player_name] = player_position
            print(f"{player_name}: {player_position}")
    
    return player_positions


def save_player_positions(player_positions, filename='player_positions.json'):
    """Save extracted positions to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(player_positions, f, indent=2, ensure_ascii=False)
    print(f"Player positions saved to {filename}")


def update_player_info_with_positions(player_positions, input_file='all_player_info_new_dates.json', 
                                    output_file='all_player_info_with_positions.json'):
    """Add position data to existing player info JSON"""
    
    # Load existing player info
    with open(input_file, 'r', encoding='utf-8') as f:
        player_info = json.load(f)
    
    # Add position data
    for player_name, position in player_positions.items():
        if player_name in player_info:
            player_info[player_name]['Position'] = position
    
    # Save updated player info
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(player_info, f, indent=2, ensure_ascii=False)
    
    print(f"Updated player info saved to {output_file}")


# Usage example:
if __name__ == "__main__":
    # Extract positions from tensor data
    player_positions = extract_player_positions_from_tensor()
    
    # Save positions to separate file
    save_player_positions(player_positions)
    
    # Update main player info file with positions
    update_player_info_with_positions(player_positions)
    
    print(f"\nFound positions for {len([p for p in player_positions.values() if p != 'Unknown'])} players")
    print(f"Unknown positions: {len([p for p in player_positions.values() if p == 'Unknown'])} players")


def get_player_position_from_data(player_name, player_positions=None):
    """Updated function to get player position from extracted data"""
    if player_positions is None:
        # Load from file if not provided
        try:
            with open('player_positions.json', 'r', encoding='utf-8') as f:
                player_positions = json.load(f)
        except FileNotFoundError:
            print("Position data not found. Run extract_player_positions_from_tensor() first.")
            return 'Unknown'
    
    return player_positions.get(player_name, 'Unknown')