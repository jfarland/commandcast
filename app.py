import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
import umap

import socket
from commandcast.data_manager import DataManager, DataManagerConfig

# Page Config
st.set_page_config(page_title="CommandCast", layout="wide", page_icon="🫡")

st.session_state.db = DataManager(
    host=socket.gethostbyname("ts_db"),
    port=9000
)

# App state (in-memory storage for uploaded data and models)
if 'datasets' not in st.session_state:
    default_df = pd.read_csv("data/m4.csv", parse_dates=True)
    default_df['ds'] = pd.to_datetime(default_df['ds'])

    st.session_state['datasets'] = {}
    st.session_state['datasets']["M4_HOURLY"] = default_df

    ds_col = 'ds'
    id_col = 'unique_id'
    measure_cols = list(default_df.drop(['ds', 'unique_id'], axis=1).columns)        

    st.session_state['dataset_config'] = DataManagerConfig(**{
        'dataset_name': 'M4_HOURLY',
        'feature_engineering': 'minimal',
        'ds_col': ds_col,
        'id_col': id_col,
        'hierarchy': [],
        'measure_cols': measure_cols
    })

    try:
        st.session_state.db.create_dataset(
            config=st.session_state.dataset_config,
            df=default_df
        )
    except:
        print("Failed to create default dataset!")

if 'models' not in st.session_state:
    st.session_state['models'] = {}

# Function to visualize time series data
def plot_all_time_series(df, ds_col, id_col, measure_cols):
    df_melted = df.melt(id_vars=[ds_col, id_col], value_vars=measure_cols,
                        var_name='variable', value_name='value')
    fig = px.line(df_melted, x=ds_col, y='value', color=id_col,
                  facet_row='variable', title='Multiple Variables over Time')
    return fig



def plot_time_series(series, title):

    fig = px.line(series, x='ds', y='values', title=title, labels={'ds': 'Date', 'values': 'Values'})

    fig.update_traces(line=dict(color='orange'))

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Y',
        xaxis=dict(showgrid=True, tickangle=45),
        yaxis=dict(showgrid=True),
        template='plotly_dark'
    )

    # Show the figure
    return fig

def plot_clusters(df):
    # Plotting the results using Plotly
    fig = px.scatter(df, x='UMAP1', y='UMAP2', color='Cluster', 
                    title='Clustering in Lower Dimensions',
                    labels={'Cluster': 'Cluster'},
                    color_continuous_scale=px.colors.qualitative.Set1)

    # Show the plot
    return fig

@st.dialog("Configure Dataset")
def configure_dataset():
    st.write("Begin by uploading historical data")
    uploaded_file = st.file_uploader("Upload Time Series Dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        st.write(df.head())
        st.session_state['datasets'][uploaded_file.name] = df
        st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
    
        dataset_name = st.text_input("Dataset Name")
        feature_engineering = st.selectbox("Feature Engineering", ["minimal", "full"])
        ds_col = st.selectbox("Timestamp Column", df.columns)
        id_col = st.selectbox("Unique ID Column", df.columns)
        hierarchy = st.multiselect("Hierarchy Columns", df.columns.tolist())
        measure_cols = st.multiselect("Measure Columns", df.columns.tolist())

        if st.button("Submit"):
            st.session_state['datasets'][uploaded_file.name + "_config"] = DataManagerConfig(**{
                "dataset_name": dataset_name,
                "feature_engineering": feature_engineering,
                "ds_col": ds_col,
                "id_col": id_col,
                "hierarchy": hierarchy,
                "measure_cols": measure_cols,
            })
            st.success("Dataset details saved successfully!")
            st.rerun()

# PAGE 1: Data Viewer
def data_viewer():
    st.title("Data Viewer")
    if st.button("Upload Data"):
        configure_dataset()

    if st.session_state['datasets']:
        selected_dataset = st.selectbox("Select Dataset", list(st.session_state['datasets'].keys()))
        df = st.session_state['datasets'][selected_dataset]
        df_config = st.session_state['dataset_config']
        with st.expander("Dataset Details..."):
            st.json(df_config.to_dict())

        st.write("Select timeseries to visualize")
        ids = df.unique_id.unique()
        selected_id = st.selectbox("Pick an ID", ids)

        if selected_id:
            series = st.session_state.db.get_series(selected_id, table_name = df_config['ts_table_name'])
            series_features = st.session_state.db.get_features(selected_id, table_name = df_config['ft_table_name'])
            st.dataframe(series_features, use_container_width=True)
            st.plotly_chart(plot_time_series(series, title=selected_id))

        if st.button("Cluster Timeseries"):
            with st.spinner("Running analysis..."):
                # retrieve timeseries features
                features = st.session_state.db.create_features(
                    df, config=df_config
                )
                feature_cols = list(features.drop(['unique_id', 'dataset_name', 'time_begin', 'time_end', 'count'], axis=1).columns)  

                # Dimensionality Reduction using UMAP
                umap_model = umap.UMAP(n_components=2, random_state=42)
                X_umap = umap_model.fit_transform(features[feature_cols])

                # Clustering using Gaussian Mixture Model
                gmm = GaussianMixture(n_components=4, random_state=42)  # Set n_components based on your data
                gmm.fit(X_umap)
                clusters = gmm.predict(X_umap)

                # Create a DataFrame for visualization
                umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
                umap_df['Cluster'] = clusters

                st.plotly_chart(plot_clusters(umap_df))

                # Filter features to include only numeric columns for calculating means
                numeric_features = features[feature_cols].select_dtypes(include=[np.number])
                cluster_means = numeric_features.groupby(clusters).mean()  # Calculate means of the features for each cluster
                
                st.dataframe(cluster_means)

                # Optionally show additional information about the clusters
                with st.expander("Cluster Details"):        
                    for cluster in range(len(cluster_means)):
                        st.write(f"**Cluster {cluster}:**")
                        st.json(cluster_means.iloc[cluster].to_dict())

# PAGE 2: Model Training
def model_training():
    st.title("Model Training")
    if not st.session_state['datasets']:
        st.warning("Please upload a dataset in the Data Viewer page.")
        return

    selected_dataset = st.selectbox("Select Dataset to Train Model", list(st.session_state['datasets'].keys()))
    dataset = st.session_state['datasets'][selected_dataset]

    #selected_column = st.selectbox("Select Time Series Column", dataset.columns)
    selected_column = "values"
    target = dataset[selected_column]

    lag = st.slider("Select Lag (number of previous steps)", 1, 10, 1)
    df_lag = dataset.copy()
    df_lag['Lag'] = target.shift(lag)
    df_lag = df_lag.dropna()

    X = df_lag['Lag'].values.reshape(-1, 1)
    y = df_lag[selected_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"Model Trained on '{selected_column}' with Lag {lag}")
    st.write(f"Test Mean Squared Error: {mse:.4f}")

    if st.button("Save Model"):
        st.session_state['models'][f"{selected_dataset}_{selected_column}_lag{lag}"] = model
        st.success(f"Model saved as '{selected_dataset}_{selected_column}_lag{lag}'")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset.index[len(y_train):], y=y_test, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=dataset.index[len(y_train):], y=y_pred, mode='lines', name='Predicted'))
    st.plotly_chart(fig)

# PAGE 3: Forecasting
def forecasting():
    st.title("Forecasting")

    if not st.session_state['models']:
        st.warning("Please train a model in the Model Training page.")
        return

    selected_model = st.selectbox("Select Trained Model", list(st.session_state['models'].keys()))
    model = st.session_state['models'][selected_model]

    selected_dataset = st.selectbox("Select Dataset for Forecasting", list(st.session_state['datasets'].keys()))
    dataset = st.session_state['datasets'][selected_dataset]

    selected_column = st.selectbox("Select Time Series Column", dataset.columns)
    target = dataset[selected_column]

    forecast_horizon = st.slider("Select Forecast Horizon", 1, 30, 5)

    lag = int(selected_model.split('_lag')[1])
    last_value = target.iloc[-lag:].values.reshape(-1, 1)

    forecast = []
    for _ in range(forecast_horizon):
        pred = model.predict(last_value)[0]
        forecast.append(pred)
        last_value = [[pred]]

    forecast_index = pd.date_range(start=dataset.index[-1], periods=forecast_horizon+1, freq='D')[1:]
    forecast_series = pd.Series(forecast, index=forecast_index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset.index, y=target, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast'))
    st.plotly_chart(fig)

# PAGE 4: Settings
def settings():
    st.title("Settings")
    st.write("Configure API keys and other settings.")
    together_api_key = st.text_input("Together.ai API Key")
    nixtla_api_key = st.text_input("NIXTLA API Key")

    # available foundational models
    foundational_models = ['timegpt', 'timesfm', 'lag-llama']
    pref_foundational_models = st.multiselect("Preferred foundational models", foundational_models, default=foundational_models[0])

    # benchmarks
    baselines = ['1-period lag', 'Historic Average', 'Seasonal Average']
    pref_baselines = st.multiselect("Preferred baselines", baselines, default=baselines[0])

    # clustering algorithms
    clustering_algos = ['GaussianMixture', 'KMeans']
    pref_clustering_algos = st.multiselect("Preferred clustering algorithms", clustering_algos, default=clustering_algos[0])
    
    # clustering algorithms
    reduction_algos = ['UMAP', 'Principal Components']
    pref_reduction_algos = st.multiselect("Preferred dimensionality reduction algorithms", reduction_algos, default=reduction_algos[0])

    if st.button("Save Settings"):
        st.session_state['api_keys'] = {
            'together_api_key': together_api_key,
            'nixtla_api_key': nixtla_api_key,
        }
        st.session_state['preferences'] = {
            'foundational_models': pref_foundational_models,
            'baselines': pref_baselines,
            'clustering_algos': pref_clustering_algos,
            'reduction_algos': pref_reduction_algos
        }
        st.success("Settings saved successfully!")

# Sidebar for navigation

st.sidebar.image("static/studio-icon.png", caption="AI Agents for Timeseries Forecasting", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["CommandCast", "Settings"])

# Page navigation
if page == "CommandCast":
    tab1, tab2, tab3 = st.tabs(["Data Viewer", "Model Training", "Forecasting"])
    with tab1:
        data_viewer()
    with tab2:
        model_training()
    with tab3:
        forecasting()
elif page == "Settings":
    settings()