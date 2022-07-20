from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

REQUIREMENT_FILE_NAME = "requirements.txt"
REMOVE_PACKAGE = "-e ."


# def get_requirement_list(requirement_file_name=REQUIREMENT_FILE_NAME) -> list:
#     try:
#         requirement_list = None
#         with open(requirement_file_name) as requirement_file:
#             requirement_list = [requirement.replace("\n", "") for requirement in requirement_file]
#             requirement_list.remove(REMOVE_PACKAGE)
#         return requirement_list
#     except Exception as e:
#         raise e



setup(
    name="census-consumer-complaint",
    license="MIT",
    version="0.0.9",
    description="Project has been completed.",
    # packages=find_packages(),
    # install_requires=['tfx==1.6.1', 'apache-beam[interactive]', 'apache-airflow']
    author="Avnish Yadav",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['tfx==1.6.1', 'apache-beam[interactive]', 'apache-airflow']
)


