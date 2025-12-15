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
    # REMOVED: Main title to save vertical space
    
    graph_df, stats_df = load_data()
    
    if graph_df is None:
        st.error("Error: Data files not found. Please run the ETL script first.")
        return

    # Sidebar Header
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

    # --- COLOR LOGIC ---
    base_colors = sns.color_palette("deep", n_colors=max(len(selected_players), 1))
    colors = list(base_colors)
    colors[0] = '#50ae26' 

    # Filter Data
    plot_df = graph_df[graph_df['Name'].isin(selected_players)]

    # --- LAYOUT CHANGE: Adjusted Split ---
    col_main, col_stats = st.columns([1.8, 1], gap="medium")

    with col_main:
        # --- Figure Setup ---
        # 1. REDUCED SIZE: (5, 6.25) is smaller but keeps the ~0.8 aspect ratio.
        # At 150 DPI, this results in an image 750px wide by 937px tall.
        fig, ax = plt.subplots(figsize=(5, 6.25), dpi=150)

        for i, player in enumerate(selected_players):
            subset = plot_df[plot_df['Name'] == player]
            if subset.empty: continue
                
            color = colors[i] if i < len(colors) else colors[-1]

            # 1. Trend Line
            ax.plot(subset['squared_up_rate'], subset['launch_angle'], color=color, 
                    linewidth=3, alpha=0.4, label=player)
            
            # 2. Bubbles
            ax.scatter(subset['squared_up_rate'], subset['launch_angle'], 
                       s=subset['obs_percentage'] * 1200, alpha=0.85, 
                       color=color, edgecolor='white', linewidth=0.75, zorder=3)

        # --- Axis Formatting ---
        ax.set_ylabel('Launch Angle (°)', fontsize=12, weight='bold')
        ax.set_xlabel('Squared-Up Rate', fontsize=12, weight='bold')
        
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(-40, 60)
        ax.set_yticks(np.arange(-40, 61, 5)) 
        
        # Dynamic X-Axis Logic
        min_rate_in_view = plot_df['squared_up_rate'].min()
        if min_rate_in_view > 0.45:
            x_start = 0.4
        else:
            x_start = max(0, min_rate_in_view - 0.05)
            
        ax.set_xlim(x_start, 1.0)
        
        # --- Static Legend ---
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        
        # Grid and Despine
        ax.grid(True, alpha=0.25)
        sns.despine()

        # Render Plot
        # 2. CRITICAL FIX: use_container_width=False prevents the graph
        # from stretching. It now strictly respects the (5, 6.25) size.
        st.pyplot(fig, use_container_width=False)

    with col_stats:
        # Spacer
        st.write("") 
        st.subheader("Stats Comparison")
        st.markdown("---")
        
        if stats_df is not None:
            display_rows = []
            
            # 1. Selected Players
            for player in selected_players:
                p_stats = stats_df[stats_df['Name'] == player]
                if not p_stats.empty:
                    row = p_stats.iloc[0].to_dict()
                    row['Type'] = 'Player'
                    display_rows.append(row)
            
            # 2. League Average
            lg_stats = stats_df[stats_df['Name'] == 'League Average']
            if not lg_stats.empty:
                row = lg_stats.iloc[0].to_dict()
                row['Type'] = 'Avg'
                display_rows.append(row)
            
            if display_rows:
                # Create Display DataFrame
                d_df = pd.DataFrame(display_rows)
                
                # Rename cols for display
                cols_to_keep = {
                    'Name': 'Name',
                    'Bat Speed': 'Bat Spd',
                    'HH%': 'HH%',
                    'GB%': 'GB%',
                    'GB SQ%': 'GB SQ%', 
                    'FB%': 'FB+', 
                    'FB SQ%': 'FB SQ%'
                }
                
                # Filter and Rename
                final_df = d_df[cols_to_keep.keys()].rename(columns=cols_to_keep)
                
                # Formatting percentages
                for col in ['HH%', 'GB%', 'GB SQ%', 'FB+', 'FB SQ%']:
                    final_df[col] = final_df[col].apply(lambda x: f"{x:.0%}")
                
                final_df['Bat Spd'] = final_df['Bat Spd'].apply(lambda x: f"{x:.1f}")
                
                # Show Dataframe
                st.dataframe(
                    final_df.set_index('Name'),
                    use_container_width=True
                )
                
                st.caption("GB = Ground Ball | FB+ = Fly Ball & Line Drive | SQ% = Squared-Up Rate on those batted balls.")

    with st.expander("View Underlying Data"):
        st.dataframe(plot_df.sort_values(by=['Name', 'launch_angle']))

if __name__ == "__main__":
    main()
