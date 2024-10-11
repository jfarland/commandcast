FROM python:3.10

# for streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit

# expose port for front end
EXPOSE 8501

WORKDIR /code
ADD requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . . 

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]