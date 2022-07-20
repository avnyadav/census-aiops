from setuptools import setup, find_packages

setup(
    name="census-consumer-complaint",
    license="MIT",
    version="0.0.7",
    description="Project has been completed.",
    author="Deepranjan",
    packages=find_packages(),
    install_requires=['tfx==1.6.1', 'apache-beam[interactive]', 'apache-airflow']
)
