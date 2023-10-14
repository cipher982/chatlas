from setuptools import find_packages, setup

setup(
    name="chatlas",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "chatlas",
        "requests",
        "pandas",
        "numpy",
        "langchain",
        "datetime",
        "openai",
        "tabulate",
        "streamlit",
    ],
)
