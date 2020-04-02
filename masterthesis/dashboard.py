from config import folder_structure
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_table
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
from sklearn import metrics

df_c = pd.read_csv(
    folder_structure.path_input + "/DashboardData.csv", index_col=0, parse_dates=True
)

df_c = df_c.dropna()
df_tmp = df_c.drop(["period", "DATE", "dataset"], axis=1)
df_tmp_2 = df_c.drop(["period", "DATE", "dataset", "future"], axis=1)


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
                                    options=get_options(df_tmp.columns),
                                    multi=True,
                                    value=["future", "H(SV)", "L(SV,40)"],
                                ),
                            ],
                        ),
                        html.P("Pick one or more measures from the Checklist below."),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="checklist",
                                    options=get_options(df_c.period.unique()),
                                    multi=False,
                                    value=df_c.period.unique()[0],
                                )
                            ],
                        ),
                        html.P("Testing Training Validation"),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="trainingselector",
                                    options=get_options(df_c.dataset.unique()),
                                    multi=False,
                                    value="testing",
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="pretty_container nine columns",
                    children=[
                        dcc.Graph(
                            id="timeseries", config={"scrollZoom": True}, animate=True,
                        ),
                    ],
                ),
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
                                    id="options_selector",
                                    options=get_options(df_tmp_2.columns),
                                    multi=True,
                                    value=["H(SV)", "L(SV,40)"],
                                ),
                            ],
                        ),
                        html.P("Pick one or more measures from the Dropdown below."),
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="quantile_selection",
                                    options=[
                                        {"label": "25th percentile", "value": 25},
                                        {"label": "50th percentile ", "value": 50},
                                        {"label": "75th percentile", "value": 75},
                                    ],
                                    multi=False,
                                    value=75,
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="pretty_container nine columns",
                    children=[
                        dcc.Graph(
                            id="violins", config={"scrollZoom": True}, animate=True,
                        )
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="pretty_container four columns",
                    children=[
                        dcc.Graph(
                            id="histogram", config={"scrollZoom": True}, animate=True,
                        )
                    ],
                ),
                html.Div(
                    className="pretty_container four columns",
                    children=[
                        dcc.Graph(
                            id="mincer", config={"scrollZoom": True}, animate=True,
                        )
                    ],
                ),
                html.Div(
                    className="pretty-container four columns",
                    children=[
                        dash_table.DataTable(
                            id="table_accuracy",
                            columns=[
                                {"name": i, "id": i}
                                for i in list(["Model", "R Squared", "MSE", "MAE"])
                            ],
                            style_cell={"textAlign": "center", "width": "16%"},
                        )
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[html.Div(className="twelve columns", children=[html.H1("")]),],
        ),
        html.Div(
            className="row",
            children=[html.Div(className="twelve columns", children=[html.H1("")]),],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(className="twelve columns bg-grey", children=[html.H1("")]),
            ],
        ),
        html.Div(
            className="row",
            children=[html.Div(className="twelve columns", children=[html.H1("")]),],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="twelve columns",
                    children=[
                        html.P(
                            " \N{COPYRIGHT SIGN} Nick Zumbühl: This dashboard has been part of the Masters Thesis for "
                            "the Master in "
                            "Quantitative Economics & Finance (MiQE/F) at the University of St. Gallen (HSG)."
                        )
                    ],
                ),
            ],
        ),
    ],
)


# Callbacks
@app.callback(
    Output("table_accuracy", "data"),
    [Input("checklist", "value"), Input("trainingselector", "value")],
)
def update_page(period_selected, data_set_selected):
    df_acc = df_c[df_c.dataset == str(data_set_selected)]
    df_acc = df_acc[df_acc.period == period_selected]

    model_names = list(df_acc.columns.drop(["dataset", "DATE", "period", "future"]))

    dict_results = {}
    for i in range(len(model_names)):
        dict_results[model_names[i]] = list(
            [
                round(metrics.r2_score(df_acc.future, df_acc[model_names[i]]), 4),
                round(
                    metrics.mean_squared_error(df_acc.future, df_acc[model_names[i]]),
                    15,
                ),
                round(
                    metrics.mean_absolute_error(df_acc.future, df_acc[model_names[i]])
                    * 100,
                    5,
                ),
            ]
        )

    df_result = pd.DataFrame(dict_results)
    df_result = df_result.transpose().reset_index()
    df_result.columns = list(["Model", "R Squared", "MSE", "MAE"])

    data = df_result.to_dict("records")

    return data


@app.callback(
    Output("timeseries", "figure"),
    [
        Input("stockselector", "value"),
        Input("checklist", "value"),
        Input("trainingselector", "value"),
    ],
)
def update_graph(selected_dropdown_value, checklist_value, data_selection):

    df_tmp = df_c[(df_c.dataset == str(data_selection))]
    df = df_tmp[df_tmp.period == checklist_value]

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
            colorway=["#004604", "#2E8C31", "#0FC5DA", "#0461C4"],
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 20, "t": 0.5, "l": 50},
            hovermode="x",
            autosize=True,
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=400,
        ),
    }

    return figure


@app.callback(
    Output("violins", "figure"),
    [
        Input("options_selector", "value"),
        Input("checklist", "value"),
        Input("trainingselector", "value"),
    ],
)
def update_graph(selected_dropdown_value, checklist_value, data_selection):

    df_tmp = df_c[(df_c.dataset == str(data_selection))]
    df = df_tmp[df_tmp.period == checklist_value]

    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Violin(
                x=pd.Series(df.shape[0] * [stock]),
                y=df[stock] - df["future"],
                name=stock,
                box_visible=True,
                points="all",
                opacity=0.8,
                meanline_visible=True,
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#004604", "#2E8C31", "#0FC5DA", "#0461C4"],
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 20, "t": 0.5, "l": 50},
            hovermode="x",
            autosize=True,
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=400,
        ),
    }

    return figure


@app.callback(
    Output("mincer", "figure"),
    [
        Input("stockselector", "value"),
        Input("checklist", "value"),
        Input("trainingselector", "value"),
    ],
)
def update_graph(selected_dropdown_value, checklist_value, data_selection):

    df_tmp = df_c[(df_c.dataset == str(data_selection))]
    df = df_tmp[df_tmp.period == checklist_value]

    trace1 = []
    for stock in selected_dropdown_value:
        if stock == "future":
            mark = "lines"
        else:
            mark = "markers"

        trace1.append(
            go.Scatter(
                x=df[stock],
                y=df["future"],
                opacity=0.5,
                name=stock,
                line={"width": 2},
                mode=mark,
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#004604", "#2E8C31", "#0FC5DA", "#0461C4"],
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 20, "t": 0.5, "l": 50},
            hovermode="x",
            autosize=True,
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=400,
        ),
    }

    return figure


@app.callback(
    Output("histogram", "figure"),
    [
        Input("stockselector", "value"),
        Input("checklist", "value"),
        Input("trainingselector", "value"),
        Input("quantile_selection", "value"),
    ],
)
def update_graph(selected_dropdown_value, checklist_value, data_selection, quant):

    df_tmp = df_c[(df_c.dataset == str(data_selection))]
    df = df_tmp[df_tmp.period == checklist_value]
    df = df[df.future <= np.percentile(df.future, quant)]

    trace1 = []
    for stock in selected_dropdown_value:
        if stock == "future":
            mark = "lines"
        else:
            mark = "markers"

        trace1.append(
            go.Scatter(
                x=df[stock],
                y=df["future"],
                opacity=0.5,
                name=stock,
                line={"width": 2},
                mode=mark,
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#004604", "#2E8C31", "#0FC5DA", "#0461C4"],
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            margin={"b": 20, "t": 0.5, "l": 50},
            hovermode="x",
            autosize=True,
            xaxis={"range": [df.DATE.min(), df.DATE.max()]},
            height=400,
        ),
    }

    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
