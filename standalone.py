#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Utilities Module
Import this file to get player position functions in your analysis scripts

Usage:
    from position_utils import get_player_position, categorize_position, load_position_data
    
    position = get_player_position("Player Name")
    category = categorize_position(position)
"""

import json
import os
from pathlib import Path

# Global variable to cache position data
_position_data_cache = None

def load_position_data(force_reload=False):
    """Load position data from JSON file with caching"""
    global _position_data_cache
    
    if _position_data_cache is not None and not force_reload:
        return _position_data_cache
    
    position_files = [
        'player_positions.json',
        'all_player_info_with_positions.json'
    ]
    
    # Try to load from position files
    for filename in position_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # If it's the player_info file, extract positions
                if filename == 'all_player_info_with_positions.json':
                    position_data = {}
                    for player_name, player_info in data.items():
                        if 'Position' in player_info:
                            position_data[player_name] = player_info['Position']
                        elif 'position' in player_info:
                            position_data[player_name] = player_info['position']
                    _position_data_cache = position_data
                else:
                    _position_data_cache = data
                
                return _position_data_cache
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print("Warning: No position data files found!")
    print("Run 'python extract_positions.py' first to create position data.")
    _position_data_cache = {}
    return _position_data_cache


def get_player_position(player_name):
    """Get player position from cached data"""
    position_data = load_position_data()
    return position_data.get(player_name, 'Unknown')


def categorize_position(position):
    """Categorize positions into detailed position categories"""
    # Direct matches first
    detailed_positions = [
        'Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
        'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
        'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
        'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
        'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
        'Right Midfield', 'Left Wing', 'Left Center Forward', 'Center Forward',
        'Right Wing', 'Right Center Forward'
    ]
    
    # Check for exact match first
    if position in detailed_positions:
        return position
    
    # Map common abbreviations and variations to detailed positions
    position_mapping = {
        # Goalkeeper variations
        'GK': 'Goalkeeper',
        
        # Defender variations
        'CB': 'Center Back',
        'LB': 'Left Back', 
        'RB': 'Right Back',
        'LWB': 'Left Wing Back',
        'RWB': 'Right Wing Back',
        'Center-Back': 'Center Back',
        'Left-Back': 'Left Back',
        'Right-Back': 'Right Back',
        'LCB': 'Left Center Back',
        'RCB': 'Right Center Back',
        
        # Midfielder variations
        'CM': 'Center Midfield',
        'CDM': 'Center Defensive Midfield',
        'CAM': 'Center Attacking Midfield',
        'LM': 'Left Midfield',
        'RM': 'Right Midfield',
        'Central Midfield': 'Center Midfield',
        'Defensive Midfield': 'Center Defensive Midfield',
        'Attacking Midfield': 'Center Attacking Midfield',
        'LAM': 'Left Attacking Midfield',
        'RAM': 'Right Attacking Midfield',
        'LDM': 'Left Defensive Midfield',
        'RDM': 'Right Defensive Midfield',
        'LCM': 'Left Center Midfield',
        'RCM': 'Right Center Midfield',
        
        # Forward variations
        'LW': 'Left Wing',
        'RW': 'Right Wing',
        'CF': 'Center Forward',
        'ST': 'Center Forward',
        'Left Winger': 'Left Wing',
        'Right Winger': 'Right Wing',
        'Striker': 'Center Forward',
        'LCF': 'Left Center Forward',
        'RCF': 'Right Center Forward'
    }
    
    # Return mapped position or Unknown
    return position_mapping.get(position, 'Unknown')


def get_position_abbreviation(position):
    """Get abbreviated form of position for display purposes"""
    return ''.join(word[0] for word in position.split())


def get_broad_position_category(position):
    """Categorize detailed positions into broad categories (GK, Def, Mid, Fwd)"""
    position = categorize_position(position)  # Ensure it's in detailed format
    
    if position == 'Goalkeeper':
        return 'Goalkeeper'
    elif position in ['Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
                     'Right Back', 'Right Center Back', 'Right Wing Back']:
        return 'Defender'
    elif position in ['Left Midfield', 'Left Attacking Midfield', 'Left Defensive Midfield', 
                     'Left Center Midfield', 'Center Defensive Midfield', 'Center Midfield', 
                     'Center Attacking Midfield', 'Right Center Midfield', 'Right Defensive Midfield', 
                     'Right Attacking Midfield', 'Right Midfield']:
        return 'Midfielder'
    elif position in ['Left Wing', 'Left Center Forward', 'Center Forward', 'Right Wing', 
                     'Right Center Forward']:
        return 'Forward'
    else:
        return 'Unknown'


def get_all_players_with_position(position=None, broad_category=None):
    """Get list of all players with specified position or category"""
    position_data = load_position_data()
    
    if position:
        return [player for player, pos in position_data.items() if pos == position]
    elif broad_category:
        return [player for player, pos in position_data.items() 
                if get_broad_position_category(pos) == broad_category]
    else:
        return list(position_data.keys())


def get_position_statistics():
    """Get statistics about position distribution"""
    position_data = load_position_data()
    
    # Count detailed positions
    detailed_counts = {}
    broad_counts = {}
    
    for player, position in position_data.items():
        # Count detailed positions
        detailed_counts[position] = detailed_counts.get(position, 0) + 1
        
        # Count broad categories
        broad_cat = get_broad_position_category(position)
        broad_counts[broad_cat] = broad_counts.get(broad_cat, 0) + 1
    
    return {
        'detailed_positions': detailed_counts,
        'broad_categories': broad_counts,
        'total_players': len(position_data),
        'unknown_players': detailed_counts.get('Unknown', 0)
    }


# List of all detailed position categories for reference
POSITION_CATEGORIES = [
    'Goalkeeper', 'Left Center Back', 'Left Back', 'Left Wing Back', 'Center Back',
    'Right Back', 'Right Center Back', 'Right Wing Back', 'Left Midfield', 
    'Left Attacking Midfield', 'Left Defensive Midfield', 'Left Center Midfield',
    'Center Defensive Midfield', 'Center Midfield', 'Center Attacking Midfield',
    'Right Center Midfield', 'Right Defensive Midfield', 'Right Attacking Midfield',
    'Right Midfield', 'Left Wing', 'Left Center Forward', 'Center Forward',
    'Right Wing', 'Right Center Forward'
]

# Abbreviated versions for plotting
POSITION_CATEGORIES_ABBR = [get_position_abbreviation(pos) for pos in POSITION_CATEGORIES]

# Broad categories
BROAD_CATEGORIES = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']


def print_position_info():
    """Print information about available position data"""
    stats = get_position_statistics()
    
    print("Position Data Summary:")
    print("=" * 40)
    print(f"Total players: {stats['total_players']}")
    print(f"Unknown positions: {stats['unknown_players']}")
    print()
    
    print("Broad Categories:")
    for category, count in stats['broad_categories'].items():
        print(f"  {category:12s}: {count:3d} players")
    print()
    
    print("Detailed Positions (top 10):")
    sorted_positions = sorted(stats['detailed_positions'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
    for position, count in sorted_positions:
        print(f"  {position:25s}: {count:3d} players")


# Test function
if __name__ == "__main__":
    print("Testing position utilities...")
    print_position_info()
    
    # Test some functions
    print("\nTesting position lookup:")
    position_data = load_position_data()
    if position_data:
        sample_player = list(position_data.keys())[0]
        pos = get_player_position(sample_player)
        broad = get_broad_position_category(pos)
        abbr = get_position_abbreviation(pos)
        
        print(f"Sample: {sample_player}")
        print(f"  Position: {pos}")
        print(f"  Broad Category: {broad}")
        print(f"  Abbreviation: {abbr}")