# Chatlas
## Description
Chatlas is a Python-based application that leverages Language Learning Models like GPT-4 to engage in smart conversations about your Google location history. The application can answer questions such as 'What is that fancy sushi place I went to last summer?' or 'How many countries did I visit in 2021?'.

## Features
- Query-based interaction with your location history
- Smart contextual understanding through GPT-4
- Easy setup and usage via Python
- Extensible architecture for adding new features or models

## Dependencies
Python 3.9+
OpenAI API Key
Google Location History JSON extract (see https://takeout.google.com/)

## Installation
```bash
# Clone the repository
git clone https://github.com/cipher982/chatlas.git

# Navigate to the project directory
cd chatlas

# Install dependencies using pip
pip install .
```

## Configuration
Obtain Google Location History API credentials and place them in credentials/location-api.json.

Obtain OpenAI API key and set it as an environment variable.
```bash
export OPENAI_API_KEY=<yourkey>
```
