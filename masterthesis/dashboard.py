from config import folder_structure
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_table
import pandas as pd
from dash.dependencies import Input, Output


df = pd.read_csv(
    folder_structure.path_input + "/DataFeatures.csv", index_col=0, parse_dates=True
)
df.DATE = df.DATE.values
df.DATE = pd.to_datetime(df.DATE, format="%Y%m%d")

df_ = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/solar.csv")


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
                        html.P("Pick one or more measures from the Dropdown below."),
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
                        html.P("Pick one or more measures from the Checklist below."),
                        html.Div(
                            children=[
                                dcc.Checklist(
                                    id="checklist",
                                    options=get_options(df.columns[1:]),
                                    value=list(df.columns[1:]),
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="pretty_container nine columns",
                    children=[
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
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="pretty_container five columns",
                    children=[
                        dcc.Graph(
                            id="histogram",
                            config={"displayModeBar": False},
                            animate=True,
                        )
                    ],
                ),
                html.Div(
                    className="pretty_container five columns offset-by-one column",
                    children=[
                        dcc.Graph(
                            id="mincer", config={"displayModeBar": False}, animate=True,
                        )
                    ],
                ),
            ],
        ),
        html.Div(
            className="twelve columns offset-by-one column",
            children=[
                dash_table.DataTable(
                    id="table_das",
                    columns=[{"name": i, "id": i} for i in df_.columns],
                    data=df_.to_dict("records"),
                )
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
                line={"width": 2},
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#31302F", "#2AA3FB", "#014678"],
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 20, "t": 0.5, "l": 50},
            hovermode="x",
            autosize=True,
            title={
                "text": "Prediction versus Realized Volatility",
                "font": {"color": "black", "size": 10},
                "x": 0,
                "pad": {"t": 100, "l": 1},
                "xanchor": "left",
                "yanchor": "top",
            },
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=300,
        ),
    }

    return figure


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
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 20, "t": 0.5, "l": 50},
            hovermode="x",
            autosize=True,
            title={
                "text": "Prediction versus Realized Volatility",
                "font": {"color": "black", "size": 10},
                "x": 0,
                "pad": {"t": 100, "l": 1},
                "xanchor": "left",
                "yanchor": "top",
            },
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=200,
        ),
    }

    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
