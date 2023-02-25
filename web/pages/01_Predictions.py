import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import pickle
from typing import Dict, List
import plotly.graph_objects as go

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[2]
MLRUNS = PROJECT_DIR / "mlruns" / "0"
HISTORIC_DATA = PROJECT_DIR / "To_somitra.xlsx"

st.set_page_config(page_title="KisaanMitra", layout="wide")


@st.cache_data
def load_historic_data(centre: str, variety: str):
    data = pd.read_excel(HISTORIC_DATA)
    data_filter = data[(data["centre"] == centre) & (data["variety"] == variety)]
    data_filter = data_filter[["date", "value"]]
    return data_filter


@st.cache_resource
def load_model(model_id: str):
    model_path = MLRUNS / model_id / "artifacts" / "model" / "model.pkl"
    if not model_path.is_file():
        st.error(
            "Model File not found for selected Centre and Variety. Please Check the Model ID is configured correctly with Streamlit app."
        )
    model = pickle.load(open(model_path, "rb"))
    return model


@st.cache_data
def load_perc_change(predicted, base):
    value = round(100 * ((predicted - base) / base), 1)
    return value


def layout_predict_wheat_price(model_id: str, centre: str, variety: str):
    centre_variety_model = load_model(model_id)
    predicted_df = centre_variety_model.predict().forecast.reset_index().round(2)
    predicted_df.columns = ["date", "value"]

    df = load_historic_data(centre, variety)
    df = pd.concat([df, predicted_df], ignore_index=True)
    df["pct_change"] = df["value"].pct_change()

    # add chart
    fig = px.line(
        df,
        x="date",
        y="value",
        range_x=["2021-01-01", "2023-04-01"],
        labels={
            "date": "Month Year",
            "value": "Price (INR/kg)",
        },
        markers=True,
    )
    fig.update_traces(
        line_color="silver",
        line_width=3,
        marker=dict(
            color="LightSlateGrey", size=7, line=dict(width=2, color="DarkSlateGrey")
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=predicted_df["date"],
            y=predicted_df["value"],
            mode="lines+markers",
            marker=dict(
                color="limegreen", size=10, line=dict(width=2, color="DarkGreen")
            ),
            hovertemplate=None,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={
            "text": f"Price Forecast for next 3 months for {centre} -  {variety}",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(
                size=25,
            ),
        },
        hovermode="x unified",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    # # add metric for current month
    # _, _, _, predicted_col, _, _ , _ = st.columns(7)
    # with predicted_col:
    #     st.metric("Predicted Price", "Rs. 2000")

    with st.container():
        # add layout metric for previous month
        _, month_1, _, month_2, _, month_3, _ = st.columns(7)

        with month_1:
            month_1_val = round(df.iloc[-3, 1])
            st.metric(
                df.iloc[-3, 0].strftime("%b-%Y"),
                month_1_val,
                f"{round(df.iloc[-3,2]*100,2)}%",
            )

        with month_2:
            month_2_val = round(df.iloc[-2, 1], 2)
            st.metric(
                df.iloc[-2, 0].strftime("%b-%Y"),
                month_2_val,
                f"{round(df.iloc[-2,2]*100,2)}%",
            )

        with month_3:
            month_3_val = round(df.iloc[-1, 1], 2)
            st.metric(
                df.iloc[-1, 0].strftime("%b-%Y"),
                month_3_val,
                f"{round(df.iloc[-1,2]*100,2)}%",
            )


label = "Select a Centre"
centre_model_mapping: Dict[str, List[str]] = {
    "Bhopal": ["564a9065bac341a28cbfedbdcc45716a", "46258876af3c452ea29c576300c22c93"],
    "Udaipur": ["5e033428b96f4a56a6ff051ee28bad08", "c55246ecd5444f68989a3f749ca71e9d"],
}

select_centre = st.selectbox(
    label,
    list(centre_model_mapping.keys()),
    index=0,
    key=list(centre_model_mapping.keys())[0],
    help=None,
    on_change=None,
    label_visibility="visible",
)

value_month_tab, explanation_tab = st.tabs(["Predict (Monthly)", "Explanations"])

with value_month_tab:
    st.title("Predict Wheat Prices")

    # add 2 columns here
    desi_wheat, high_yield_wheat = st.columns(2)

    with st.expander("Desi Wheat"):
        layout_predict_wheat_price(
            model_id=centre_model_mapping[select_centre][0],
            centre=select_centre,
            variety="Desi",
        )

    with st.expander("Kalyan HYV"):
        layout_predict_wheat_price(
            model_id=centre_model_mapping[select_centre][1],
            centre=select_centre,
            variety="Kalyan HYV",
        )

with explanation_tab:
    st.title("Explanations for the Wheat Prices")
