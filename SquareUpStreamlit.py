# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:10:12 2025

@author: benja
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Page Config
st.set_page_config(page_title="Squared-Up Rate Explorer", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    try:
        # Load Graph Data AND Summary Stats
        graph_df = pd.read_csv('squared_up_precalc.csv')
        stats_df = pd.read_csv('player_summary_stats.csv')
        return graph_df, stats_df
    except FileNotFoundError:
        return None, None

def main():
    st.title("⚾ Squared-Up Rate by Launch Angle")
    st.markdown("Compare hitter performance relative to their theoretical maximum exit velocity.")

    graph_df, stats_df = load_data()
    
    if graph_df is None:
        st.error("Error: Data files not found. Please run the ETL script first.")
        return

    # Sidebar
    st.sidebar.header("Configuration")
    
    # Get unique list of players
    available_players = sorted(graph_df['Name'].unique())
    
    selected_players = st.sidebar.multiselect(
        "Select Players",
        options=available_players,
        default=available_players[:1] if available_players else None,
        help="Type to search."
    )

    if not selected_players:
        st.info("Select a player to begin.")
        return

    # Filter Data for Graph
    plot_df = graph_df[graph_df['Name'].isin(selected_players)]
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Color Logic
    # Force FanGraphs Green for first player, use 'deep' palette for others
    base_colors = sns.color_palette("deep", n_colors=max(len(selected_players), 1))
    colors = list(base_colors)
    colors[0] = '#50ae26' 

    # Containers for Table Data
    table_cell_text = []
    table_rows = []
    table_colors = []

    for i, player in enumerate(selected_players):
        # 1. Plot Graph
        subset = plot_df[plot_df['Name'] == player]
        if subset.empty: 
            continue
            
        color = colors[i] if i < len(colors) else colors[-1]

        # Trend Line
        ax.plot(
            subset['launch_angle'], 
            subset['squared_up_rate'], 
            color=color, 
            linewidth=3, 
            alpha=0.4, 
            label=player
        )
        
        # Bubbles
        ax.scatter(
            subset['launch_angle'], 
            subset['squared_up_rate'], 
            s=subset['obs_percentage'] * 1000, 
            alpha=0.85, 
            color=color, 
            edgecolor='white', 
            linewidth=0.75, 
            zorder=3
        )
        
        # 2. Prepare Table Data
        # Get stats for this player from the stats_df
        if stats_df is not None:
            p_stats = stats_df[stats_df['Name'] == player]
            if not p_stats.empty:
                p_stats = p_stats.iloc[0]
                
                # Format: "45% (80%)"
                gb_str = f"{p_stats['GB%']:.1%} ({p_stats['GB SQ%']:.1%})"
                fb_str = f"{p_stats['FB%']:.1%} ({p_stats['FB SQ%']:.1%})"
                
                table_cell_text.append([gb_str, fb_str])
                table_rows.append(player)
                table_colors.append(color)

    # Axis Formatting
    ax.set_xlabel('Launch Angle (°)', fontsize=12, weight='bold')
    ax.set_ylabel('Squared-Up Rate', fontsize=12, weight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-40, 60)
    ax.set_xticks(np.arange(-40, 61, 10))
    
    # Legend (Top Right)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.25)
    sns.despine()

    # --- Add Data Table (Bottom Right) ---
    if table_cell_text:
        col_labels = ["GB% (SQ%)", "FB+ (SQ%)"]
        
        # Add the table to the plot
        the_table = plt.table(
            cellText=table_cell_text,
            rowLabels=table_rows,
            colLabels=col_labels,
            rowColours=table_colors,
            cellLoc='center',
            loc='bottom right',
            # bbox: [x, y, width, height] relative to axes
            # Dynamically adjust height based on number of players
            bbox=[0.65, 0.02, 0.33, 0.15 + (0.03 * len(selected_players))] 
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.5) # Add some padding to cells

    # RENDER WITH CONTAINER WIDTH
    st.pyplot(fig, use_container_width=True)

    with st.expander("View Underlying Data"):
        st.dataframe(plot_df.sort_values(by=['Name', 'launch_angle']))

if __name__ == "__main__":
    main()