
## How to setup airflow

Set airflow directory
```
export AIRFLOW_HOME="/home/avnish/census_consumer_project/census_consumer_complaint/airflow"
```

To install airflow 
```
pip install apache-airflow
```

To configure databse
```
airflow db init
```

To create login user for airflow
```
airflow users create  -e avnish@ineuron.ai -f Avnish -l Yadav -p admin -r Admin  -u admin
```
To start scheduler
```
airflow scheduler
```
To launch airflow server
```
airflow webserver -p <port_number>
```

```
pip install pandas-tfrecords
```

```
pip install \
  --upgrade --ignore-installed \
  python-snappy==0.5.1 \
  --global-option=build_ext \
  --global-option="-I/usr/local/include" \
  --global-option="-L/usr/local/lib"
```

pip install twine
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
twine upload dist/*

