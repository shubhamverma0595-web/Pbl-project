"""
SmartHealth AI - Exploratory Data Analysis (EDA) Script
======================================================
Generates correlation heatmap, feature distribution plots,
and class balance summaries for PBL analysis and reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "..", "dataset", "heart_disease.csv")
    output_dir = os.path.join(base_dir, "..", "eda_plots")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    print("--- Dataset Statistical Summary ---")
    print(df.describe())

    # Set aesthetic style
    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("viridis")

    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Heart Disease Clinical Features Correlation Matrix", fontsize=14, fontweight='bold')
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")

    # 2. Target Class Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="target", palette=["#2ecc71", "#e74c3c"])
    plt.title("Class Balance: Normal vs Heart Disease Presence", fontsize=12, fontweight='bold')
    plt.xlabel("Target (0 = Normal, 1 = Disease Present)")
    plt.ylabel("Patient Count")
    class_dist_path = os.path.join(output_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to: {class_dist_path}")

    # 3. Age vs Max Heart Rate Scatter Plot by Target
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="age", y="thalach", hue="target", palette=["#3498db", "#e74c3c"], alpha=0.8, s=70)
    plt.title("Age vs Maximum Heart Rate (thalach) by Disease Status", fontsize=12, fontweight='bold')
    plt.xlabel("Age (years)")
    plt.ylabel("Max Heart Rate Achieved (bpm)")
    scatter_path = os.path.join(output_dir, "age_vs_thalach.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot to: {scatter_path}")

    print("\nEDA completed successfully! Plots saved in 'eda_plots/' directory.")


if __name__ == "__main__":
    perform_eda()
