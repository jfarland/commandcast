import pandas as pd 
from pandas import json_normalize
import numpy as np

from tqdm import tqdm

# AutoGen Agentic Framework
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, register_function
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor

from typing_extensions import Annotated

from .model_manager import ModelManager

import streamlit as st
import tempfile

### PLATFORM AGENTS ###

# example: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_web_info.ipynb

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

class UserProxy(UserProxyAgent):
    def __init__(self, llm_config, streamlit=True):
        super().__init__(
            name="user_proxy",
            human_input_mode="ALWAYS",
            system_message="You are a helpful assistant which takes inputs from a human user.",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            llm_config = llm_config)
        self.streamlit = streamlit

    #override the process received message for streamlit
    def _process_received_message(self, message, sender, silent):
        if self.streamlit:
            with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message

class CodeExecutor(UserProxyAgent):
    # this agent uses docker for code execution
    def __init__(self, streamlit=True):
        # Create a temporary directory to store the code files.
        self.work_dir = tempfile.TemporaryDirectory()

        # Create a Docker command line code executor.
        self.executor = DockerCommandLineCodeExecutor(
            image="python:3.12-slim",  # Execute code using the given docker image name.
            timeout=30,  # Timeout for each code execution in seconds.
            work_dir=self.work_dir.name,  # Use the temporary directory to store the code files.
        )

        super().__init__(
            name='code_executor',
            code_execution_config={"executor": self.executor},
            human_input_mode="NEVER"  # Use the docker command line code executor.
        )

class StatsForecaster(AssistantAgent):
    # this agent focuses on 
    def __init__(self, llm_config, streamlit=True):
        super().__init__(
            name='stats_forecaster',
            human_input_mode="NEVER", # user never talks to this agent directly
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=3,
            llm_config=llm_config
        )
        self.streamlit = streamlit

    #override the process received message for streamlit
    def _process_received_message(self, message, sender, silent):
        if self.streamlit:
            with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message
 
class NeuralForecaster(AssistantAgent):
    # this agent focuses on 
    def __init__(self, llm_config, streamlit=True):
        super().__init__(
            name='stats_forecaster',
            human_input_mode="NEVER", # user never talks to this agent directly
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=3,
            llm_config=llm_config
        )
        self.streamlit = streamlit

    #override the process received message for streamlit
    def _process_received_message(self, message, sender, silent):
        if self.streamlit:
            with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message

class FoundationForecaster(AssistantAgent):
    # this agent focuses on 
    def __init__(self, llm_config, streamlit=True):
        super().__init__(
            name='stats_forecaster',
            human_input_mode="NEVER", # user never talks to this agent directly
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=3,
            llm_config=llm_config
        )
        self.streamlit = streamlit

    #override the process received message for streamlit
    def _process_received_message(self, message, sender, silent):
        if self.streamlit:
            with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message

class Reconciler(AssistantAgent):
    # this agent focuses on 
    def __init__(self, llm_config, streamlit=True):
        super().__init__(
            name='stats_forecaster',
            human_input_mode="NEVER", # user never talks to this agent directly
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=3,
            llm_config=llm_config
        )
        self.streamlit = streamlit

    #override the process received message for streamlit
    def _process_received_message(self, message, sender, silent):
        if self.streamlit:
            with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message

class Combiner(AssistantAgent):
    # this agent focuses on 
    def __init__(self, llm_config, streamlit=True):
        super().__init__(
            name='stats_forecaster',
            human_input_mode="NEVER", # user never talks to this agent directly
            default_auto_reply="Reply `TERMINATE` if the task is done.",
            max_consecutive_auto_reply=3,
            llm_config=llm_config
        )
        self.streamlit = streamlit

    #override the process received message for streamlit
    def _process_received_message(self, message, sender, silent):
        if self.streamlit:
            with st.chat_message(sender.name):
                st.markdown(message)
        return super()._process_received_message
