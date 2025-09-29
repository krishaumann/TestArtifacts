import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import os
import glob

EXPORT_FOLDER = "Exports"

for csv_path in glob.glob(os.path.join(EXPORT_FOLDER, "*.csv")):
        print(f"Processing file: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

# Build the full model once
G_full = nx.DiGraph()
node_areas = {}

for idx, row in df.iterrows():
    area = row["area_name"]
    paths_str = str(row["consolidated_paths"])
    if not paths_str or pd.isna(paths_str):
        continue
    paths = [p.strip() for p in paths_str.split(" | ") if p.strip()]
    for path in paths:
        steps = path.split(" -> ")
        for i in range(len(steps) - 1):
            G_full.add_edge(steps[i], steps[i + 1], area=area)
        for step in steps:
            node_areas.setdefault(step, set()).add(area)

area_list = sorted(set(df["area_name"].dropna()))
color_palette = plt.cm.get_cmap('tab20', len(area_list))
area_color_map = {area: color_palette(i) for i, area in enumerate(area_list)}

def show_model(selected_indices):
    filter_areas = [area_list[i] for i in selected_indices]
    if not filter_areas:
        G = G_full
        title = "Consolidated Test Model (All Functional Areas)"
        legend_areas = area_list
    else:
        G = nx.DiGraph()
        for idx, row in df.iterrows():
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

    node_colors = []
    for node in G.nodes():
        areas = node_areas.get(node, [])
        if areas:
            if filter_areas:
                area = sorted([a for a in areas if a in filter_areas])[0] if any(a in filter_areas for a in areas) else sorted(areas)[0]
            else:
                area = sorted(areas)[0]
            node_colors.append(area_color_map.get(area, "grey"))
        else:
            node_colors.append("grey")

    num_nodes = len(G.nodes())
    fig_width = max(16, num_nodes // 2)
    fig_height = max(10, num_nodes // 3)
    font_size = max(8, 20 - num_nodes // 10)
    node_size = max(800, 2000 - num_nodes * 10)

    plt.figure(figsize=(fig_width, fig_height))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=font_size)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray', width=2)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=area_color_map[area], label=area) for area in legend_areas]
    plt.legend(handles=legend_handles, title="Functional Areas", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    components = list(nx.weakly_connected_components(G))
    messagebox.showinfo("Info", f"Number of connected components: {len(components)}")

def on_show():
    selected_indices = listbox.curselection()
    show_model(selected_indices)

root = tk.Tk()
root.title("Functional Area Filter")

tk.Label(root, text="Select functional areas (Ctrl+Click for multi-select):").pack()

listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=20)
for area in area_list:
    listbox.insert(tk.END, area)
listbox.pack()

show_btn = tk.Button(root, text="Show Model", command=on_show)
show_btn.pack(pady=10)

root.mainloop()