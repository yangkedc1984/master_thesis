from config import folder_structure
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output


df = pd.read_csv(
    folder_structure.path_input + "/DataFeatures.csv", index_col=0, parse_dates=True
)
df.DATE = df.DATE.values
df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")


def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({"label": i, "value": i})

    return dict_list


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


app.layout = html.Div(
    children=[
        html.H1(
            className="title bg-grey",
            children=[
                "LSTM Neural Networks and HAR Models for Realized Volatility - "
                "An Application to Financial Volatility Forecasting"
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="pretty_container three columns",
                    children=[
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="stockselector",
                                    options=get_options(df.columns[1:]),
                                    multi=True,
                                    value=list(df.columns[1:]),
                                ),
                            ],
                        ),
                        html.P("Pick one or more measures from the dropdown below."),
                        html.Div(
                            children=[
                                dcc.Checklist(
                                    id="checklist",
                                    options=get_options(df.columns[1:]),
                                    value=list(["RV"]),
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="pretty_container nine columns",
                    children=[
                        # html.H3("Result Time Series One"),
                        dcc.Graph(
                            id="timeseries",
                            config={"displayModeBar": False},
                            animate=True,
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.P(
                    className="pretty_container three columns",
                    children=["Add another checkbox here"],
                ),
                html.Div(
                    className="pretty_container nine columns",
                    children=[
                        dcc.Graph(
                            id="updatechange",
                            config={"displayModeBar": False},
                            animate=True,
                        )
                    ],
                ),
            ],
        ),
    ],
)


# Callback for timeseries price
@app.callback(Output("timeseries", "figure"), [Input("stockselector", "value")])
def update_graph(selected_dropdown_value):
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(
                x=df.DATE,
                y=df[stock],
                mode="lines",
                opacity=0.7,
                name=stock,
                #  textposition="bottom center",
                line={"width": 2},
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#31302F", "#2AA3FB", "#014678"],
            # template="plotly_dark",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 1},
            hovermode="x",
            autosize=False,
            # title={
            #     "text": "Different Realized Volatility Measures",
            #     "font": {"color": "white"},
            #     "x": 0.5,
            # },
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=400,
        ),
    }

    return figure


#'height': 400,
# 'margin': {'l': 10, 'b': 20, 't': 0, 'r': 0}


@app.callback(Output("updatechange", "figure"), [Input("checklist", "value")])
def update_change(selection_checklist):
    trace1 = []
    for stock in selection_checklist:
        trace1.append(
            go.Scatter(
                x=df.DATE,
                y=df[stock],
                mode="lines",
                opacity=0.7,
                name=stock,
                # textposition="bottom center",
                line={"width": 2},
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#31302F", "#2AA3FB", "#014678"],
            # template="plotly_dark",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 1},
            hovermode="x",
            autosize=True,
            # title={
            #     "text": "Different Realized Volatility Measures 2.0",
            #     "font": {"color": "white"},
            #     "x": 0.5,
            # },
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=400,
        ),
    }

    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
