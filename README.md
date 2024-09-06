# REGEX-DRIVEN INFORMATION EXTRACTION AND TEXT SUMMARIZATION

![image](https://github.com/user-attachments/assets/cfafbf76-d252-42a6-a2ad-7742c664def1)

## Project Description

This project focuses on **information extraction using regular expressions (regex)** and **text summarization** with the help of **K-Means clustering**. The application integrates regex-based extraction techniques to efficiently retrieve key information from text and uses a pre-trained **Word2Vec model** to generate text embeddings, which are then clustered and summarized using **K-Means**.

The application is built using **Flask** for the backend and is designed as a web application where users can upload text or documents for analysis.

## Project Features

- **Information Extraction**: The system extracts key entities such as dates, names, emails, phone numbers, etc., from input text using regular expressions.
- **Text Summarization**: Using **K-Means clustering**, the project generates summarized versions of large text bodies.
- **Pre-trained Word2Vec Model**: The text embeddings are generated using a pre-trained Word2Vec model trained on Indonesian vocabulary.
- **Web Application**: The application provides a user-friendly interface for users to upload files and view the extracted information and text summary.

## Folder Structure

```
regex-app/
├── app/
│   ├── models/
│   │   ├── (Word2Vec models will be stored here)
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   └── app.py               # Main Flask application script
├── venv/                    # Virtual environment (not included in the repo)
└── requirements.txt         # Dependencies list
```

## Prerequisites

- Python 3.x
- Virtual environment (`venv`)

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/naufalraihasa/regex-driven-info-extraction.git
cd regex-driven-info-extraction
```

### 2. Create a Virtual Environment

Set up a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows**: 
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**: 
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Word2Vec Models

Since the Word2Vec model files are too large to push to GitHub, you can download the pre-trained models from the following Google Drive link:

- [Download Word2Vec models from Google Drive](https://drive.google.com/drive/folders/1WN87_f8bbxMPl7QDQYNLYXxe9j1lvsgN?usp=sharing)

Once downloaded, create a folder called `models` inside the `app` directory and place the downloaded model files there:

```
TUGAS 1/
├── app/
│   ├── models/
│   │   ├── word2vec-indonesia.model
│   │   ├── word2vec-indonesia.model.syn1neg.npy
│   │   └── word2vec-indonesia.model.wv.vectors.npy
...
```

### 5. Run the Flask Application

Now, you can run the Flask application locally:

```bash
python app.py
```

The app will be accessible at `http://127.0.0.1:5000/` in your web browser.

## Usage

1. Open the app in a browser at `http://127.0.0.1:5000/`.
2. Upload a text file or input some text into the provided input area.
3. The application will extract important information using regex and display the summary generated through text summarization and clustering.

## License

This project is licensed under the MIT License.

---

### Notes:
- The pre-trained **Word2Vec** model is not included in this repository due to file size constraints. Please download the models as instructed above.
- For any issues with setting up the project, feel free to create an issue on GitHub.

---