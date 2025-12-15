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
import matplotlib.patheffects as path_effects # Added for text outlines

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

    # --- COLOR LOGIC ---
    # We define colors early so we can use them in the Sidebar Legend AND the plot
    base_colors = sns.color_palette("deep", n_colors=max(len(selected_players), 1))
    colors = list(base_colors)
    colors[0] = '#50ae26' # Override first color to green
    
    # --- FLOATING LEGEND (IN SIDEBAR) ---
    # Since the sidebar is sticky, this legend will "float" with the user
    st.sidebar.markdown("---")
    st.sidebar.subheader("Legend")
    for player, color in zip(selected_players, colors):
        # We use a little HTML/CSS to make a colored dot
        st.sidebar.markdown(
            f"<span style='color:{color}; font-size:1.2em;'>●</span> **{player}**", 
            unsafe_allow_html=True
        )

    # Filter Data for Graph
    plot_df = graph_df[graph_df['Name'].isin(selected_players)]

    # --- LAYOUT: Split Screen ---
    col_main, col_stats = st.columns([3, 1], gap="medium")

    with col_main:
        # Taller Figure (Portrait)
        fig, ax = plt.subplots(figsize=(9, 12), dpi=150)

        for i, player in enumerate(selected_players):
            subset = plot_df[plot_df['Name'] == player]
            if subset.empty: continue
                
            color = colors[i] if i < len(colors) else colors[-1]

            # 1. Trend Line
            ax.plot(subset['squared_up_rate'], subset['launch_angle'], color=color, 
                    linewidth=3, alpha=0.4) # Removed label here, handling manually
            
            # 2. Bubbles
            ax.scatter(subset['squared_up_rate'], subset['launch_angle'], 
                       s=subset['obs_percentage'] * 1200, alpha=0.85, 
                       color=color, edgecolor='white', linewidth=0.75, zorder=3)
            
            # 3. INLINE LABELS (Top and Bottom)
            # Find point with Max Launch Angle (Top)
            top_point = subset.loc[subset['launch_angle'].idxmax()]
            # Find point with Min Launch Angle (Bottom)
            bot_point = subset.loc[subset['launch_angle'].idxmin()]
            
            # Helper to draw text with white halo (readable against grid)
            def add_label(row, text_offset_y=0):
                txt = ax.text(
                    row['squared_up_rate'], row['launch_angle'] + text_offset_y, 
                    f"  {player}", 
                    color=color, fontsize=10, weight='bold', va='center'
                )
                txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

            add_label(top_point, text_offset_y=0)
            
            # Optional: Label bottom too if graph is very tall
            if abs(top_point['launch_angle'] - bot_point['launch_angle']) > 30:
                 add_label(bot_point, text_offset_y=0)

        # --- Axis Formatting ---
        ax.set_ylabel('Launch Angle (°)', fontsize=14, weight='bold')
        ax.set_xlabel('Squared-Up Rate', fontsize=14, weight='bold')
        
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(-40, 60)
        ax.set_yticks(np.arange(-40, 61, 5))
        
        # Dynamic X-Axis
        min_rate_in_view = plot_df['squared_up_rate'].min()
        x_start = 0.4 if min_rate_in_view > 0.45 else max(0, min_rate_in_view - 0.05)
        ax.set_xlim(x_start, 1.0)
        
        # We removed the standard legend ax.legend() in favor of Sidebar + Inline
        ax.grid(True, alpha=0.25)
        sns.despine()

        st.pyplot(fig, use_container_width=True)

    with col_stats:
        st.subheader("Stats Comparison")
        st.markdown("---")
        
        if stats_df is not None:
            display_rows = []
            
            # 1. Selected Players
            for player in selected_players:
                p_stats = stats_df[stats_df['Name'] == player]
                if not p_stats.empty:
                    row = p_stats.iloc[0].to_dict()
                    display_rows.append(row)
            
            # 2. League Average
            lg_stats = stats_df[stats_df['Name'] == 'League Average']
            if not lg_stats.empty:
                row = lg_stats.iloc[0].to_dict()
                display_rows.append(row)
            
            if display_rows:
                d_df = pd.DataFrame(display_rows)
                
                cols_to_keep = {
                    'Name': 'Name',
                    'Bat Speed': 'Bat Spd',
                    'HH%': 'HH%',
                    'GB%': 'GB%',
                    'GB SQ%': 'GB SQ%', 
                    'FB%': 'FB+', 
                    'FB SQ%': 'FB SQ%'
                }
                
                final_df = d_df[cols_to_keep.keys()].rename(columns=cols_to_keep)
                
                # Format
                for col in ['HH%', 'GB%', 'GB SQ%', 'FB+', 'FB SQ%']:
                    final_df[col] = final_df[col].apply(lambda x: f"{x:.0%}")
                final_df['Bat Spd'] = final_df['Bat Spd'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(final_df.set_index('Name'), use_container_width=True)
                st.caption("GB = Ground Ball | FB+ = Fly Ball & Line Drive | SQ% = Squared-Up Rate on those batted balls.")

    with st.expander("View Underlying Data"):
        st.dataframe(plot_df.sort_values(by=['Name', 'launch_angle']))

if __name__ == "__main__":
    main()
