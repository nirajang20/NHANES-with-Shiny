from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
from faicons import icon_svg
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Folder structure ---
DATA_FOLDERS = ["15-16", "17-18", "21-23"]
DATA_FILES = {
    "HEI": "HEI_*filtered.csv",
    "MED": "MED_*filtered.csv",
    "MIND": "MIND_*filtered.csv",
}

# --- UI ---
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Choose Dataset"),
        ui.input_select("cycle", "Cycle (Year Range):", DATA_FOLDERS, selected="15-16"),
        ui.input_select("diet_type", "Diet Type:", list(DATA_FILES.keys()), selected="HEI"),
        ui.input_action_button("load_btn", "Load Dataset", class_="btn-primary"),
        ui.hr(),
        ui.h4("Configure Plot"),
        ui.input_select(
            "plot_type",
            "Plot Type:",
            {"scatter": "Scatter Plot", "regression": "Regression Plot", "histogram": "Histogram", "box": "Box Plot"},
            selected="scatter",
        ),
        ui.input_select("x", "X-axis:", choices=[]),
        ui.panel_conditional(
            "['scatter', 'regression', 'box'].includes(input.plot_type)",
            ui.input_select("y", "Y-axis:", choices=[]),
        ),
        ui.panel_conditional(
            "input.plot_type == 'histogram'",
            ui.input_slider("hist_bins", "Histogram bins:", min=5, max=100, value=30),
        ),
        ui.input_select("color", "Color / Group By:", choices=[]),
    ),
    ui.h2(icon_svg("chart-line"), " NHANES Nutrition Score Dashboard"),
    ui.h3(icon_svg("table"), " Data Preview"),
    ui.output_table("table_preview"),
    ui.hr(),
    ui.h3(icon_svg("chart-area"), " Interactive Plot"),
    output_widget("plot"),
)

# --- Server ---
def server(input, output, session):
    data_meta = reactive.Value({"numeric_cols": [], "categorical_cols": []})

    @reactive.calc
    @reactive.event(input.load_btn)
    def load_data():
        folder = input.cycle()
        dtype = input.diet_type()

        # Find file dynamically
        path = None
        for f in os.listdir(folder):
            if dtype in f and f.endswith(".csv"):
                path = os.path.join(folder, f)
                break

        if path is None:
            raise FileNotFoundError(f"No file found for {dtype} in {folder}")

        df = pd.read_csv(path, low_memory=False)

        if not df.empty:
            # Many NHANES exports keep a metadata row directly under the header.
            first_row = df.iloc[0].dropna()
            alpha_frac = 0.0
            if not first_row.empty:
                alpha_frac = first_row.astype(str).str.contains(r"[A-Za-z]", na=False).mean()
            if alpha_frac > 0.5:
                df = df.iloc[1:].reset_index(drop=True)

        # Drop empty rows/columns prior to type coercion
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        if "SEQN" in numeric_df.columns:
            df["SEQN"] = numeric_df["SEQN"].round().astype("Int64")
            numeric_df["SEQN"] = df["SEQN"]

        numeric_cols = [
            col
            for col in numeric_df.columns
            if pd.api.types.is_numeric_dtype(numeric_df[col]) and numeric_df[col].dropna().shape[0] > 0
        ]
        categorical_cols = [
            col
            for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().shape[0] > 0
        ]

        if not numeric_cols:
            raise ValueError(f"The dataset for {dtype} in {folder} does not contain any numeric columns to plot.")

        # Cache column metadata for downstream use
        data_meta.set({
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        })

        # Update axis selectors dynamically
        x_selected = numeric_cols[0]
        y_selected = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]

        ui.update_select("x", choices=numeric_cols, selected=x_selected, session=session)
        ui.update_select("y", choices=numeric_cols, selected=y_selected, session=session)

        color_choices = ["None"] + [col for col in numeric_cols + categorical_cols if col]
        # Preserve ordering while removing duplicates
        seen = set()
        deduped_color = []
        for col in color_choices:
            if col not in seen:
                seen.add(col)
                deduped_color.append(col)

        ui.update_select("color", choices=deduped_color, selected="None", session=session)

        return df

    @output
    @render.table
    def table_preview():
        df = load_data()
        return df.head(10)

    @output
    @render_widget
    def plot():
        df = load_data()
        meta = data_meta.get()
        numeric_cols = meta.get("numeric_cols", [])

        plot_type = input.plot_type()
        x_col = input.x()
        y_col = input.y()

        color_value = input.color()
        color_col = None if color_value in (None, "None") else color_value

        if plot_type in {"scatter", "regression"} and len(numeric_cols) < 2:
            raise ValueError("Select a dataset with at least two numeric columns to create this plot.")

        required_columns = []
        required_numeric = []

        if plot_type in {"scatter", "regression"}:
            if not x_col or not y_col:
                raise ValueError("Select both X and Y columns for scatter or regression plots.")
            required_columns.extend([x_col, y_col])
            required_numeric.extend([x_col, y_col])
        elif plot_type == "histogram":
            if not x_col:
                raise ValueError("Select an X column for the histogram.")
            required_columns.append(x_col)
            required_numeric.append(x_col)
        elif plot_type == "box":
            if not y_col:
                raise ValueError("Select a Y column for the box plot.")
            required_columns.append(y_col)
            required_numeric.append(y_col)
            if x_col:
                required_columns.append(x_col)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        if color_col:
            required_columns.append(color_col)

        unique_required_cols = []
        for col in required_columns:
            if col and col not in unique_required_cols:
                unique_required_cols.append(col)

        missing_cols = [col for col in unique_required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The selected columns are not available in the dataset: {', '.join(missing_cols)}.")

        plot_df = df[unique_required_cols].copy()

        # Ensure numeric columns are in numeric form
        for col in {c for c in required_numeric if c}:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
            if plot_df[col].dropna().shape[0] == 0:
                raise ValueError(f"The selected column '{col}' does not contain numeric data to plot.")

        numeric_subset = [col for col in required_numeric if col]
        plot_df = plot_df.dropna(subset=numeric_subset)

        if plot_df.empty:
            raise ValueError("No rows remain after filtering out missing values for the selected columns.")

        # Remove rows with missing grouping variables to keep Plotly happy
        if plot_type == "box" and x_col:
            plot_df = plot_df.dropna(subset=[x_col])
        if color_col:
            plot_df = plot_df.dropna(subset=[color_col])

        if plot_type == "scatter":
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                template="plotly_white",
            )
        elif plot_type == "regression":
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                template="plotly_white",
            )

            trend_df = plot_df.sort_values(by=x_col)
            if trend_df[x_col].nunique() < 2:
                raise ValueError("Need at least two distinct X values to compute a regression line.")

            x_values = trend_df[x_col].to_numpy(dtype=float)
            y_values = trend_df[y_col].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x_values, y_values, 1)
            line_x = np.linspace(x_values.min(), x_values.max(), 100)
            line_y = slope * line_x + intercept

            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="Trend line",
                    line=dict(color="#d62728"),
                )
            )
        elif plot_type == "histogram":
            bins = input.hist_bins()
            fig = px.histogram(
                plot_df,
                x=x_col,
                color=color_col,
                nbins=bins,
                template="plotly_white",
            )
            fig.update_layout(bargap=0.05)
        elif plot_type == "box":
            box_kwargs = {"y": y_col, "template": "plotly_white"}
            if x_col and x_col in plot_df.columns:
                box_kwargs["x"] = plot_df[x_col]
            if color_col:
                box_kwargs["color"] = color_col

            fig = px.box(plot_df, **box_kwargs)

        fig.update_layout(height=500)
        return fig


app = App(app_ui, server)
