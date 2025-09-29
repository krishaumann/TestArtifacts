"""
============================================================
Script: Functional Area Test Model Visualizer
============================================================

📌 Purpose
-----------
This script loads consolidated test paths from CSV exports,
builds a directed graph of functional area flows, and provides
an interactive GUI for filtering and visualizing the model.

The visualization helps QA/Test managers to:
- Understand how different test steps connect across functional areas.
- Quickly see overlaps and dependencies between modules.
- Identify disconnected components in the test model.

✨ Main Features
----------------
1. **CSV Loader**
   - Reads the latest CSV from the `Exports/` folder.
   - Validates presence of required columns: `area_name`, `consolidated_paths`.

2. **Graph Builder**
   - Builds a full directed graph from all test paths.
   - Tracks which functional areas each node belongs to.
   - Creates a color-coded mapping of functional areas.

3. **Visualization**
   - Interactive Tkinter GUI to select one or more functional areas.
   - Uses NetworkX and Matplotlib to render the graph dynamically.
   - Scales figure size, node size, and font size based on graph size.
   - Adds a color legend for functional areas.

4. **Connectivity Insights**
   - After rendering, shows the number of weakly connected components.
   - Helps to identify isolated or disconnected test flows.

5. **Error Handling**
   - Validates CSV existence and required columns.
   - Catches and reports visualization errors via messageboxes.

👤 Usage
---------
- Place CSV files in the `Exports/` folder.
- Run this script: `python test_model_visualizer.py`
- Select functional areas in the GUI listbox and click "Show Model".

Dependencies
-------------
- pandas
- networkx
- matplotlib
- tkinter (built-in with Python)

============================================================
"""

import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.patches import Patch

# ============================================================
# Configuration
# ============================================================
EXPORT_FOLDER = "Exports"

# (rest of the script continues unchanged…)

import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.patches import Patch

# ============================================================
# Configuration
# ============================================================
EXPORT_FOLDER = "Exports"


# ============================================================
# Load Data
# ============================================================
def load_latest_csv(export_folder: str) -> pd.DataFrame:
    """
    Load the first CSV file found in the given export folder.

    Args:
        export_folder (str): Path to the folder containing CSV exports.

    Returns:
        pd.DataFrame: Loaded DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If no CSV files are found in the folder.
        ValueError: If the CSV file cannot be read.
        KeyError: If required columns are missing.
    """
    csv_files = glob.glob(os.path.join(export_folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {export_folder}")

    # Just take the first one found (could be extended to allow selection)
    csv_path = csv_files[0]
    print(f"Processing file: {csv_path}")

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Failed to read CSV {csv_path}: {e}")

    # Ensure required columns exist
    for col in ["area_name", "consolidated_paths"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing in CSV {csv_path}")

    return df


# ============================================================
# Graph Building
# ============================================================
def build_full_graph(df: pd.DataFrame) -> tuple[nx.DiGraph, dict]:
    """
    Build the consolidated graph from all functional areas.

    Args:
        df (pd.DataFrame): DataFrame containing 'area_name' and 'consolidated_paths'.

    Returns:
        tuple:
            - nx.DiGraph: Directed graph of all paths across all areas.
            - dict: Mapping of node -> set of areas the node belongs to.
    """
    G = nx.DiGraph()
    node_areas: dict[str, set] = {}

    for _, row in df.iterrows():
        area = row["area_name"]
        paths_str = str(row["consolidated_paths"])

        if not paths_str or pd.isna(paths_str):
            continue

        # Split multiple path strings separated by "|"
        paths = [p.strip() for p in paths_str.split(" | ") if p.strip()]
        for path in paths:
            steps = path.split(" -> ")

            # Add edges between steps
            for i in range(len(steps) - 1):
                G.add_edge(steps[i], steps[i + 1], area=area)

            # Track which areas each node belongs to
            for step in steps:
                node_areas.setdefault(step, set()).add(area)

    return G, node_areas


def build_area_color_map(area_list: list[str]) -> dict:
    """
    Assign a unique color to each functional area.

    Args:
        area_list (list[str]): List of distinct area names.

    Returns:
        dict: Mapping area -> RGBA color.
    """
    color_palette = plt.cm.get_cmap("tab20", len(area_list))
    return {area: color_palette(i) for i, area in enumerate(area_list)}


# ============================================================
# Visualization
# ============================================================
def show_model(selected_indices: list[int]) -> None:
    """
    Display a graph (full or filtered) based on the user's selection.

    Args:
        selected_indices (list[int]): Listbox indices of selected areas.

    Returns:
        None. Shows a matplotlib graph window and a Tkinter messagebox with summary info.
    """
    filter_areas = [area_list[i] for i in selected_indices]

    # ------------------------------------------------------------
    # Step 1: Build Graph
    # ------------------------------------------------------------
    if not filter_areas:
        # Use the prebuilt full graph
        G = G_full
        title = "Consolidated Test Model (All Functional Areas)"
        legend_areas = area_list
    else:
        # Build a filtered graph with only selected areas
        G = nx.DiGraph()
        for _, row in df.iterrows():
            area = row["area_name"]
            if area not in filter_areas:
                continue

            paths_str = str(row["consolidated_paths"])
            if not paths_str or pd.isna(paths_str):
                continue

            paths = [p.strip() for p in paths_str.split(" | ") if p.strip()]
            for path in paths:
                steps = path.split(" -> ")
                for i in range(len(steps) - 1):
                    G.add_edge(steps[i], steps[i + 1], area=area)

        title = f"Test Model for Functional Areas: {', '.join(filter_areas)}"
        legend_areas = filter_areas

    # ------------------------------------------------------------
    # Step 2: Color Nodes
    # ------------------------------------------------------------
    node_colors = []
    for node in G.nodes():
        areas = node_areas.get(node, [])
        if areas:
            if filter_areas:
                # Prefer one of the selected areas if overlaps
                area = (
                    sorted([a for a in areas if a in filter_areas])[0]
                    if any(a in filter_areas for a in areas)
                    else sorted(areas)[0]
                )
            else:
                area = sorted(areas)[0]
            node_colors.append(area_color_map.get(area, "grey"))
        else:
            node_colors.append("grey")

    # ------------------------------------------------------------
    # Step 3: Dynamic Layout Sizing
    # ------------------------------------------------------------
    num_nodes = len(G.nodes())
    fig_width = max(16, num_nodes // 2)
    fig_height = max(10, num_nodes // 3)
    font_size = max(8, 20 - num_nodes // 10)
    node_size = max(800, 2000 - num_nodes * 10)

    # ------------------------------------------------------------
    # Step 4: Draw Graph
    # ------------------------------------------------------------
    plt.figure(figsize=(fig_width, fig_height))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle="-|>", arrowsize=20, edge_color="gray", width=2
    )

    # Add legend
    legend_handles = [Patch(color=area_color_map[area], label=area) for area in legend_areas]
    plt.legend(handles=legend_handles, title="Functional Areas", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Step 5: Show Info
    # ------------------------------------------------------------
    components = list(nx.weakly_connected_components(G))
    messagebox.showinfo("Info", f"Number of connected components: {len(components)}")


# ============================================================
# GUI Logic
# ============================================================
def on_show() -> None:
    """
    Callback for the 'Show Model' button.
    Fetches selected indices and displays the corresponding graph.

    Returns:
        None. Either shows a graph or an error messagebox.
    """
    selected_indices = listbox.curselection()
    try:
        show_model(selected_indices)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate model: {e}")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    try:
        # Load CSV data
        df = load_latest_csv(EXPORT_FOLDER)

        # Build full graph and color map
        G_full, node_areas = build_full_graph(df)
        area_list = sorted(set(df["area_name"].dropna()))
        area_color_map = build_area_color_map(area_list)
    except Exception as e:
        print(f"Startup failed: {e}")
        raise SystemExit(1)

    # ----------------- GUI -----------------
    root = tk.Tk()
    root.title("Functional Area Filter")

    # Label
    tk.Label(root, text="Select functional areas (Ctrl+Click for multi-select):").pack()

    # Multi-select listbox
    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=20)
    for area in area_list:
        listbox.insert(tk.END, area)
    listbox.pack()

    # Show button
    show_btn = tk.Button(root, text="Show Model", command=on_show)
    show_btn.pack(pady=10)

    # Run app
    root.mainloop()
