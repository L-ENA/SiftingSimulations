import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_results_summary(csv_path: str) -> pd.DataFrame:
    """Load the results summary CSV.

    Expected columns:
    - dataset
    - run
    - work_saved_bias_1, true_recall_bias_1
    - work_saved_bias_1.5, true_recall_bias_1.5
    - work_saved_bias_2, true_recall_bias_2
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"results_summary file not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide bias-specific columns into a long format with one row per
    dataset / run / bias combination, and add per-dataset means for each bias.
    """
    bias_configs = [
        ("work_saved_bias_1", "true_recall_bias_1", "1.0"),
        ("work_saved_bias_1.5", "true_recall_bias_1.5", "1.5"),
        ("work_saved_bias_2", "true_recall_bias_2", "2.0"),
    ]

    records = []

    # Per-run rows (use iterrows so column names with dots are preserved)
    for _, row in df.iterrows():
        for work_col, recall_col, bias_label in bias_configs:
            records.append(
                {
                    "dataset": row["dataset"],
                    "run": row["run"],
                    "bias": bias_label,
                    "work_saved": row[work_col],
                    "true_recall": row[recall_col],
                    "type": "run",
                }
            )

    # Per-dataset means
    for dataset, group in df.groupby("dataset"):
        for work_col, recall_col, bias_label in bias_configs:
            mean_work = group[work_col].mean()
            mean_recall = group[recall_col].mean()
            records.append(
                {
                    "dataset": dataset,
                    # Use a descriptive pseudo-run label for means
                    "run": "mean",
                    "bias": bias_label,
                    "work_saved": mean_work,
                    "true_recall": mean_recall,
                    "type": "mean",
                }
            )

    return pd.DataFrame.from_records(records)


def plot_all_runs(long_df: pd.DataFrame, output_html: str | None = None) -> None:
    """Create a scatter plot showing all runs and per-dataset means for each
    bias parameter in a single figure.
    """
    title = (
        "Early Stopping at 95% estimated recall: "
        "Records not needed to be screened vs. True underlying recall (all datasets)"
    )

    fig = px.scatter(
        long_df,
        x="work_saved",
        y="true_recall",
        color="bias",
        symbol="type",
        symbol_map={"run": "circle", "mean": "diamond"},
        hover_data=["dataset", "run", "bias", "type"],
        title=title,
        labels={
            "work_saved": "Percentage of data not needed to be seen",
            "true_recall": "True underlying recall",
            "bias": "Bias parameter",
        },
        template="simple_white",
    )

    # Horizontal reference line at true recall = 0.95
    if not long_df.empty:
        x_min = 0.0
        x_max = max(long_df["work_saved"].max(), 0.1)
        fig.add_scatter(
            x=[x_min, x_max],
            y=[0.95, 0.95],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Recall 0.95",
        )

    # Style markers
    fig.update_traces(marker_line_color="black")

    # Make dataset-average points (type == 'mean') larger
    for trace in fig.data:
        # Skip non-marker traces, e.g. the horizontal reference line
        if not hasattr(trace, "marker"):
            continue
        symbol = getattr(trace.marker, "symbol", None)
        if symbol == "diamond":
            trace.marker.size = 15

    # Connect mean points (diamonds) for each dataset with a thin black line
    means_df = long_df[long_df["type"] == "mean"].copy()
    if not means_df.empty:
        # Ensure a consistent ordering of bias values along the line
        bias_order = ["1.0", "1.5", "2.0"]
        means_df["bias"] = pd.Categorical(means_df["bias"], categories=bias_order, ordered=True)

        for dataset, group in means_df.groupby("dataset"):
            group_sorted = group.sort_values("bias")
            fig.add_scatter(
                x=group_sorted["work_saved"],
                y=group_sorted["true_recall"],
                mode="lines",
                line=dict(color="black", width=1),
                name=f"{dataset} mean trajectory",
                showlegend=False,
            )

    fig.show()

    if output_html is None:
        output_html = os.path.abspath("all_runs_summary.html")

    fig.write_html(output_html)


def main() -> None:
    """Run the plot using fixed input/output paths defined in this script."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust these paths if your folder layout changes.
    csv_path = os.path.join(base_dir, "data", "gaincurves", "buscar", "results_summary_hs_data.csv")
    output_html = os.path.join(base_dir, "data", "gaincurves", "buscar", "all_runs_summary.html")

    df = load_results_summary(csv_path)
    long_df = build_long_format(df)
    plot_all_runs(long_df, output_html)


if __name__ == "__main__":
    main()
