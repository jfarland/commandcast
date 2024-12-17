FROM python:3.10

###### uv install ######
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/0.5.10/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
###### uv install ######

# for streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit

# expose port for front end
EXPOSE 8501

WORKDIR /code
ADD pyproject.toml .
ADD uv.lock .
RUN uv sync --frozen
COPY . . 
# using 'uv run' so streamlit will be ran in the same virtualenv,   
ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]