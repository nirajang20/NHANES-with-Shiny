from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
from faicons import icon_svg
import pandas as pd
import plotly.express as px
import os

# --- Folder structure ---
DATA_FOLDERS = ["15-16", "17-18", "21-23"]
DATA_FILES = {
    "HEI": "HEI_*filtered.csv",
    "MED": "MED_*filtered.csv",
    "MIND": "MIND_*filtered.csv",
}

# --- UI ---
app_ui = ui.page_fluid(
    ui.h2(icon_svg("chart-line") + " NHANES Nutrition Score Dashboard"),

    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.h4("Choose Dataset"),
            ui.input_select("cycle", "Cycle (Year Range):", DATA_FOLDERS, selected="15-16"),
            ui.input_select("diet_type", "Diet Type:", list(DATA_FILES.keys()), selected="HEI"),
            ui.input_action_button("load_btn", "Load Dataset", class_="btn-primary"),
            ui.hr(),
            ui.input_select("x", "X-axis (for Plot):", choices=[]),
            ui.input_select("y", "Y-axis (for Plot):", choices=[]),
            ui.input_select("color", "Color By:", choices=[]),
        ),
        ui.panel_main(
            ui.h3(icon_svg("table") + " Data Preview"),
            ui.output_table("table_preview"),
            ui.hr(),
            ui.h3(icon_svg("chart-area") + " Interactive Plot"),
            output_widget("plot"),
        )
    )
)

# --- Server ---
def server(input, output, session):

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

        df = pd.read_csv(path)
        # Drop empty columns if any
        df = df.dropna(axis=1, how='all')

        # Update axis selectors dynamically
        session.send_input_message("x", {"choices": list(df.columns), "selected": list(df.columns)[0]})
        session.send_input_message("y", {"choices": list(df.columns), "selected": list(df.columns)[1]})
        session.send_input_message("color", {"choices": ["None"] + list(df.columns), "selected": "None"})

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
        color_col = None if input.color() == "None" else input.color()
        fig = px.scatter(df, x=input.x(), y=input.y(), color=color_col, template="plotly_white")
        fig.update_layout(height=500)
        return fig


app = App(app_ui, server)
