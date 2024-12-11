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

    def __init__(self, host, port, ingress_port=9009):
        self.host = host
        self.api_port = port
        self.ingress_port = ingress_port
        self.db_host = f'http://{host}:{port}'

        # Ensure datasets table exists
        sql_query = 'CREATE TABLE IF NOT EXISTS data_catalog (name STRING, timeseries_table STRING, features_table STRING, ingest_timestamp TIMESTAMP);'
        self.run(sql_query)

    def run(self, sql_query):
        query_params = {'query': sql_query, 'fmt': 'json'}
        try:
            response = requests.get(self.db_host + '/exec', params=query_params).json()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error: {e}")
            return None

    def extract_config(self, config):
        """Extracts and transforms config for internal use"""
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

    def count_table_records(self, table_name):
        sql = f"""SELECT COUNT(*) FROM {table_name};"""
        result = self.run(sql)
        return result['dataset'][0][0]

    def check_table_exists(self, table_name):
        sql = f"""SELECT * FROM tables WHERE table_name = '{table_name}';"""
        result = self.run(sql)
        
        # Check if the result contains any rows (i.e., the table exists)
        if result and "dataset" in result:
            return len(result["dataset"]) > 0
        return False

    def create_dataset(self, config, df):
        """Creates dataset by writing to timeseries records and features"""
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

    def record_dataset(self, config):
        """Inserts dataset information into the data_catalog table."""
        sql_query = f"""
            INSERT INTO data_catalog (name, timeseries_table, features_table, ingest_timestamp) 
            VALUES ('{config['dataset_name']}', '{config['ts_table_name']}', '{config['ft_table_name']}', now());
        """
        self.run(sql_query)

    def load_features(self, df, config, verbose=True):
        """Prepares and load engineered features."""
        try:
            features = self.create_features(df, config)
            if verbose:
                logger.info(features.head())
            self.ingest_data(features, config['ft_table_name'], ['unique_id', 'dataset_name'], 'time_begin')

            return True
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return False

    def load_dataset(self, df, config):
        """Loads timeseries data."""
        try:
            self.ingest_data(df, config['ts_table_name'], ['unique_id'], 'ds')
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False

    def ingest_data(self, df, table_name, symbols, time_col):
        """Ingests data into the specified table using Influx Line Protocol."""
        try:
            with Sender('tcp', self.host, self.ingress_port) as sender:
                buf = sender.new_buffer()
                buf.dataframe(df, table_name=table_name, symbols=symbols, at=time_col)
                sender.flush(buf)
        except IngressError as e:
            logger.error(f"Ingress error: {e}")
            raise

    def create_features(self, df, config):
        """Creates engineered features based on the provided dataframe and config."""
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

    def get_series(self, unique_id, table_name):
        ''' Function to retrieve a time series by its unique_id'''
        
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
        
    def get_features(self, unique_id, table_name):
        ''' Function to retrieve features by its unique_id'''
        
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
