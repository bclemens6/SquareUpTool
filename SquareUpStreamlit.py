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
    
    available_players = sorted([x for x in graph_df['Name'].unique() if x != 'League Average'])
    
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
    
    # Setup Figure (14x6 Widescreen)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # Color Logic
    base_colors = sns.color_palette("deep", n_colors=max(len(selected_players), 1))
    colors = list(base_colors)
    colors[0] = '#50ae26' 

    # Containers for Table Data
    table_cell_text = []
    table_rows = []
    table_colors = []

    # --- 1. Process Selected Players ---
    for i, player in enumerate(selected_players):
        subset = plot_df[plot_df['Name'] == player]
        if subset.empty: continue
            
        color = colors[i] if i < len(colors) else colors[-1]

        # Trend Line (X=Squared Up, Y=Launch Angle)
        ax.plot(subset['squared_up_rate'], subset['launch_angle'], color=color, 
                linewidth=3, alpha=0.4, label=player)
        
        # Bubbles
        ax.scatter(subset['squared_up_rate'], subset['launch_angle'], 
                   s=subset['obs_percentage'] * 1200, alpha=0.85, 
                   color=color, edgecolor='white', linewidth=0.75, zorder=3)
        
        # Get Stats for Table
        if stats_df is not None:
            p_stats = stats_df[stats_df['Name'] == player]
            if not p_stats.empty:
                p_stats = p_stats.iloc[0]
                
                bat_spd = f"{p_stats['Bat Speed']:.1f}"
                hh_rate = f"{p_stats['HH%']:.1%}"
                gb_str = f"{p_stats['GB%']:.0%} ({p_stats['GB SQ%']:.0%})"
                fb_str = f"{p_stats['FB%']:.0%} ({p_stats['FB SQ%']:.0%})"
                
                table_cell_text.append([bat_spd, hh_rate, gb_str, fb_str])
                table_rows.append(player)
                table_colors.append(color)

    # --- 2. Add League Average Row ---
    if stats_df is not None:
        lg_stats = stats_df[stats_df['Name'] == 'League Average']
        if not lg_stats.empty:
            lg = lg_stats.iloc[0]
            
            lg_bat_spd = f"{lg['Bat Speed']:.1f}"
            lg_hh_rate = f"{lg['HH%']:.1%}"
            lg_gb_str = f"{lg['GB%']:.0%} ({lg['GB SQ%']:.0%})"
            lg_fb_str = f"{lg['FB%']:.0%} ({lg['FB SQ%']:.0%})"
            
            table_cell_text.append([lg_bat_spd, lg_hh_rate, lg_gb_str, lg_fb_str])
            table_rows.append("League Avg")
            table_colors.append("#e0e0e0")

    # --- Axis Formatting ---
    ax.set_ylabel('Launch Angle (°)', fontsize=14, weight='bold')
    ax.set_xlabel('Squared-Up Rate', fontsize=14, weight='bold')
    
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(-40, 60)
    ax.set_yticks(np.arange(-40, 61, 10))
    
    # --- Dynamic X-Axis Logic ---
    # Find the lowest squared-up rate in the current view
    min_rate_in_view = plot_df['squared_up_rate'].min()
    
    # If the lowest data point is safe (e.g., > 45%), cut the graph at 40%
    # Otherwise, give a small buffer (0.05) below the minimum, floored at 0
    if min_rate_in_view > 0.45:
        x_start = 0.4
    else:
        x_start = max(0, min_rate_in_view - 0.05)
        
    ax.set_xlim(x_start, 1.0)
    
    # --- Legend (Upper Left) ---
    ax.legend(loc='upper left', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.25)
    sns.despine()

    # --- Data Table (Bottom Left) ---
    if table_cell_text:
        col_labels = ["Bat Spd (All Swings)", "HH%", "GB% (SQ%)", "FB+ (SQ%)"]
        
        the_table = plt.table(
            cellText=table_cell_text,
            rowLabels=table_rows,
            colLabels=col_labels,
            rowColours=table_colors,
            cellLoc='center',
            loc='bottom left', # Anchor to bottom left
            # bbox: [x, y, width, height] relative to axes
            # Positioned at x=0.02 (left side), y=0.02 (bottom)
            bbox=[0.02, 0.02, 0.45, 0.15 + (0.04 * len(table_rows))] 
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.5) 

    st.pyplot(fig, use_container_width=True)

    with st.expander("View Underlying Data"):
        st.dataframe(plot_df.sort_values(by=['Name', 'launch_angle']))

if __name__ == "__main__":
    main()
