from setuptools import setup, find_packages

setup(
    name="chatlas",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "langchain",
        # "dotenv",
        "datetime",
    ],
)
