
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



# To deploy your model
```
pip install tensorflow-serving-api
```
to inspect model

```
saved_model_cli show --dir <dir_path>
```
Above command will return tag set
```commandline
saved_model_cli show --dir <dir_path> --tag_set <tag_name>
```
Above command will show available model signatures

Next: with tag_set and signature_def info, 
we can inspect model input and output
```commandline
saved_model_cli show --dir <dir_path> --tag_set <tag_name> --signature_def <SignatureDef Key>
```


To inspect all signature without tag_set and signature_def
saved_model_cli show --dir <dir_path> --all


Testing the model
```commandline

```
Test model prediction using saved_model_cli with sample input data

>--input_examples: input data formatted as a tf.Example data structure

### other param
> --outdir: by default output will be written in terminal

> --overwrite: to write into a file

> tf_debug: run in debug mode

To expose your model as an API using docker image tensorflow/serving
```
docker pull tensorflow/serving
```

## Single model configuration
```
sudo docker run -p 8500:8500 \
-p 8501:8501\
--volumn <model_dir>:<target_dir>\
-e MODEL_NAME=<model_name>\
-e model_base_path=<target_dir>\
-t tensorflow/serving:latest
```

```
sudo docker run -p 8500:8500 -p 8501:8501 \
-v  /home/avnish/census_consumer_project/census_consumer_complaint/census_consumer_complaint_data/saved_models:/avnish/my_model \
-e MODEL_NAME=my_model \
-e MODEL_BASE_PATH=/avnish \
-t tensorflow/serving:latest
```

To inspect docker container directory
```commandline
docker exec -it <conatiner_name> bash
```