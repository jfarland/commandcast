# data wrangling
import pandas as pd
import numpy as np

# os ops
import sys
import requests
import os 

# other utilities
from tqdm import tqdm
import uuid
from distutils.util import strtobool 
import requests
from loguru import logger

# timeseries feature engineering
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

# database connection
from questdb.ingress import Sender, IngressError
import questdb.ingress


class DataManager:
    """
    The DataManager class is responsible for managing interactions with a time-series database. 

    It supports operations like creating datasets, extracting features, loading data, 
    and querying time-series and feature tables. The class integrates data engineering 
    functionalities with a focus on time-series data storage and retrieval.

    Attributes:
        host (str): The database host.
        api_port (int): The API port for database interactions.
        ingress_port (int): The port for data ingestion using the QuestDB Ingress API.
        db_host (str): The full URL of the database host for API queries.
    """

    def __init__(self, host: str, port: int, ingress_port: int = 9009) -> None:
        """
        Initializes the DataManager instance with database connection details.

        Args:
            host (str): The hostname or IP address of the database server.
            port (int): The port number for the database's API endpoint.
            ingress_port (int, optional): The port number for the data ingestion interface. Defaults to 9009.

        Sets up the database connection URL and ensures that the `data_catalog` table exists for tracking datasets.
        """
        self.host = host
        self.api_port = port
        self.ingress_port = ingress_port
        self.db_host = f'http://{host}:{port}'

        # Ensure datasets table exists
        sql_query = 'CREATE TABLE IF NOT EXISTS data_catalog (name STRING, timeseries_table STRING, features_table STRING, ingest_timestamp TIMESTAMP);'
        self.run(sql_query)

    def run(self, sql_query: str) -> dict | None:
        """
        Executes a SQL query on the database and retrieves the result.

        Args:
            sql_query (str): The SQL query string to be executed.

        Returns:
            dict | None: The JSON response from the database if the query is successful.
                         None if an error occurs during the query execution.
        """
        query_params = {'query': sql_query, 'fmt': 'json'}
        try:
            response = requests.get(self.db_host + '/exec', params=query_params).json()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error: {e}")
            return None

    def extract_config(self, config: dict) -> dict:
        """Extracts and transforms configuration details for internal use.

        Args:
            config (dict): A dictionary containing dataset configuration.

        Returns:
            dict: Transformed configuration with dataset and table details.
        """
        dataset_name = config['dataset_name']
        return {
            'dataset_name': dataset_name,
            'ts_table_name': f"{dataset_name}_ts",
            'ft_table_name': f"{dataset_name}_ft",
            'hierarchy': config['hierarchy'],
            'measure_cols': config['measure_cols'],
            'id_col': 'unique_id',
            'ds_col': 'ds',
            'feature_engineering': config.get('feature_engineering', 'minimal'),
        }

    def count_table_records(self, table_name: str) -> int:
        """Counts the number of records in a specified table.

        Args:
            table_name (str): The name of the table to query.

        Returns:
            int: The number of records in the table.
        """
        sql = f"""SELECT COUNT(*) FROM {table_name};"""
        result = self.run(sql)
        return result['dataset'][0][0]

    def check_table_exists(self, table_name: str) -> bool:
        """Checks if a specified table exists in the database.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        sql = f"""SELECT * FROM tables WHERE table_name = '{table_name}';"""
        result = self.run(sql)
        
        # Check if the result contains any rows (i.e., the table exists)
        if result and "dataset" in result:
            return len(result["dataset"]) > 0
        return False

    def create_dataset(self, config: dict, df: pd.DataFrame) -> None:
        """Creates a dataset by writing timeseries records and features to the database.

        Args:
            config (dict): Configuration details for the dataset.
            df (pd.DataFrame): DataFrame containing the data to be ingested.

        Returns:
            None
        """
        new_config = self.extract_config(config)

        if self.check_table_exists(new_config['ft_table_name']) and self.count_table_records(new_config['ts_table_name']):
            logger.warning("Timeseries and features table already exists. Aborting ingest.")
            return 
        else:
            try:
                logger.info("Creating dataset...")
                ft_status = self.load_features(df, new_config)
                ts_status = self.load_dataset(df, new_config)

                if ft_status and ts_status:
                    self.record_dataset(new_config)
                else:
                    logger.error("Dataset creation failed.")
            except Exception as e:
                logger.error(f"Error during dataset creation: {e}")

    def record_dataset(self, config: dict) -> None:
        """Inserts dataset information into the data catalog.

        Args:
            config (dict): Configuration details including dataset and table names.

        Returns:
            None
        """
        sql_query = f"""
            INSERT INTO data_catalog (name, timeseries_table, features_table, ingest_timestamp) 
            VALUES ('{config['dataset_name']}', '{config['ts_table_name']}', '{config['ft_table_name']}', now());
        """
        self.run(sql_query)

    def load_features(self, df: pd.DataFrame, config: dict, verbose: bool = True) -> bool:
        """Prepares and loads engineered features into the database.

        Args:
            df (pd.DataFrame): DataFrame containing raw data.
            config (dict): Configuration for feature generation.
            verbose (bool): Flag to enable logging of feature details.

        Returns:
            bool: True if features were successfully loaded, False otherwise.
        """
        try:
            features = self.create_features(df, config)
            if verbose:
                logger.info(features.head())
            self.ingest_data(features, config['ft_table_name'], ['unique_id', 'dataset_name'], 'time_begin')

            return True
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return False

    def load_dataset(self, df: pd.DataFrame, config: dict) -> bool:
        """Loads timeseries data into the database.

        Args:
            df (pd.DataFrame): DataFrame containing timeseries data.
            config (dict): Configuration for dataset ingestion.

        Returns:
            bool: True if dataset was successfully loaded, False otherwise.
        """
        try:
            self.ingest_data(df, config['ts_table_name'], ['unique_id'], 'ds')
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def ingest_data(self, df: pd.DataFrame, table_name: str, symbols: list[str], time_col: str) -> None:
        """Ingests data into a specified table using Influx Line Protocol.

        Args:
            df (pd.DataFrame): DataFrame containing the data to be ingested.
            table_name (str): Name of the table to ingest data into.
            symbols (list[str]): List of symbol columns.
            time_col (str): Name of the time column.

        Returns:
            None
        """
        try:
            with Sender('tcp', self.host, self.ingress_port) as sender:
                buf = sender.new_buffer()
                buf.dataframe(df, table_name=table_name, symbols=symbols, at=time_col)
                sender.flush(buf)
        except IngressError as e:
            logger.error(f"Ingress error: {e}")
            raise

    def create_features(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Creates engineered features based on the provided DataFrame and configuration.

        Args:
            df (pd.DataFrame): DataFrame containing the raw data.
            config (dict): Configuration specifying feature engineering details.

        Returns:
            pd.DataFrame: A DataFrame containing engineered features.
        """
        base_features = df.groupby(config['id_col'])[config['ds_col']].agg(['min', 'max', 'count']).reset_index()
        base_features['dataset_name'] = config['dataset_name']
        
        if config['feature_engineering'] == 'minimal':
            settings = MinimalFCParameters()
        else:
            settings = EfficientFCParameters() if config['feature_engineering'] == 'efficient' else None
        
        if settings:
            extracted_features = extract_features(df, column_id=config['id_col'], column_sort=config['ds_col'], default_fc_parameters=settings)
            # Assign the current index to a new column 'unique_id'
            extracted_features['unique_id'] = extracted_features.index

            # Reset the index without inserting it into the DataFrame
            extracted_features.reset_index(drop=True, inplace=True)
            base_features = pd.merge(base_features, extracted_features.dropna(axis=1), on=config['id_col'], how='left')
        
        base_features.rename(columns={"min": "time_begin", "max": "time_end"}, inplace=True)
        return base_features

    def get_series(self, unique_id: str, table_name: str) -> pd.DataFrame | None:
        """Retrieves a time series record from the specified table by its unique identifier.

        Args:
            unique_id (str): The unique identifier for the time series.
            table_name (str): The name of the table containing the time series data.

        Returns:
            pd.DataFrame | None: A DataFrame containing the time series data, or None if retrieval fails.
        """
        try:
            sql_query = f"SELECT * FROM {table_name} WHERE unique_id = '{unique_id}';"
            response = self.run(sql_query)
        
            columns = pd.json_normalize(response, 'columns')
            
            df_parsed = pd.json_normalize(response, 'dataset')
            df_parsed.columns = columns['name'].values 
            
            if "timestamp" in df_parsed.columns:
                df_parsed.rename(columns={"timestamp": "ds"}, inplace=True)   
                    
            return df_parsed
        except Exception as e:
            logger.error(f"Failed to retrieve time series: {e}")
            return None
        
    def get_features(self, unique_id: str, table_name: str) -> pd.DataFrame | None:
        """Retrieves features for a specific unique identifier from a specified table.

        Args:
            unique_id (str): Unique identifier for the features to retrieve.
            table_name (str): Name of the table containing the features.

        Returns:
            pd.DataFrame | None: A DataFrame containing the retrieved features.
        """
        try:
            sql_query = f"SELECT * FROM {table_name} WHERE unique_id = '{unique_id}';"
            response = self.run(sql_query)
        
            columns = pd.json_normalize(response, 'columns')
            
            df_parsed = pd.json_normalize(response, 'dataset')
            df_parsed.columns = columns['name'].values 
                    
            return df_parsed
        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            return None
