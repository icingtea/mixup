FROM python:3.13
WORKDIR /usr/local/app

COPY r.txt ./

EXPOSE 8000

RUN pip install -r r.txt

COPY src/infer/infer.py ./

RUN useradd app
USER app

CMD ["python", "infer.py"]