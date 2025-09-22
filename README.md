# Aryan_miimansa_assingment
An NLP project for medical entity recognition on patient posts using a Hugging Face and gemini LLM. Links entities to SNOMED CT codes and evaluates string matching against semantic search via embeddings.

# Medical Entity Recognition and Linking on the CADEC Dataset

This project implements a complete Natural Language Processing (NLP) pipeline to automatically extract, label, and link medical entities from patient forum posts. The primary goal is to identify **Adverse Drug Reactions (ADRs)**, **Drugs**, **Symptoms**, and **Diseases**, and then link the identified ADRs to standardized medical concepts using SNOMED CT (`sct`).

The project uses the **CADEC dataset**, which contains patient forum posts and detailed annotations. The NER task is performed using Google's Gemini API, and the entity linking task compares a traditional string-matching approach with a modern semantic search approach using sentence embeddings.

***

## 📋 Key Features

* **Data Parsing & Analysis**: Scripts to load, parse, and analyze the complex annotation structure of the CADEC dataset.
* **Named Entity Recognition (NER)**: Identifies medical entities from raw text using the Google Gemini API, structuring the output in the required format.
* **Comprehensive Performance Evaluation**: Measures model performance using **Precision, Recall, and F1-Score** against two different ground truth sources (`original` and `meddra` annotations).
* **Entity Linking**: Links identified text segments (ADRs) to their standard SNOMED CT (`sct`) descriptions using two distinct methods:
    1.  Approximate String Matching (`fuzzywuzzy`)
    2.  Semantic Search with Hugging Face Embeddings (`sentence-transformers`)
* **Comparative Analysis**: Evaluates and compares the effectiveness of the string matching and semantic embedding approaches for the entity linking task.

***

##  Workflow

The project is broken down into a series of steps, each handled by a dedicated Jupyter Notebook.

1.  **Initial Data Exploration (`Assingment_1_Aryan.ipynb`)**
    * Loads and parses the `.ann` files from the `original`, `meddra`, and `sct` directories.
    * Converts the primary `original` annotation files into a Pandas DataFrame.
    * Performs an initial analysis to enumerate the distinct entities of each label type (ADR, Drug, Disease, Symptom) and calculate their frequencies across the dataset.

2.  **Named Entity Recognition (`Assingment_2_and_5_Aryan.ipynb`)**
    * Reads the raw text files from the `text` directory.
    * Utilizes the `gemini-1.5-flash-latest` model to perform NER on the text, prompted to extract entities and format them as per the ground truth specification.
    * *Note: This approach was chosen due to hardware limitations that prevented the effective use of larger, locally-hosted Hugging Face models.*
    * The model's output is parsed and saved to a new directory (`original2`) for evaluation.

3.  **Performance Evaluation (`Assingment_3_and_5_Aryan.ipynb` & `Assingment_4_Aryan.ipynb`)**
    * The model's predictions (from `original2`) are compared against the ground truth annotations.
    * `Assignment_3` evaluates performance against the general annotations in the `original` directory.
    * `Assignment_4` performs a more focused evaluation, comparing only the `ADR` labels against the `meddra` directory annotations.
    * Performance is measured using **Precision, Recall, and F1-Score** on a word-by-word basis.

4.  **Entity Linking & Comparison (`Assingment_6_Aryan.ipynb`)**
    * A ground truth DataFrame is created by merging data from the `original` and `sct` directories to map text segments to their official `sct_name`.
    * The predicted `ADR` entities from the model are then linked to this ground truth using two methods:
        * **Approximate String Matching**: Uses the `fuzzywuzzy` library to find the closest `sct_name` based on string similarity.
        * **Semantic Embedding Search**: Uses the `all-MiniLM-L6-v2` model from `sentence-transformers` to generate embeddings for both the predicted text and the ground truth `sct` text. It then matches entities based on cosine similarity.
    * Finally, the accuracy of both linking methods is calculated and compared.

***

## 📜 Results

### Entity Linking Performance

The primary goal of the final task was to compare the effectiveness of string matching versus semantic search for linking predicted ADRs to their standard `sct_name`. The results clearly show the superiority of the embedding-based approach.

| Method                       | Custom Accuracy |
| ---------------------------- | --------------- |
| Approximate String Matching  | 53.85%          |
| **Semantic Embedding Search** | **69.23%** |

The semantic search model was able to correctly link more entities, including those where the phrasing was different but the meaning was the same (e.g., "muscle/joint pain" to "Arthralgia").

### NER Model Performance

Performance metrics were calculated for the first 5 files as a sample.

**General Performance (vs. `original` folder)**

| File          | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| ARTHROTEC.1   | 1.000     | 1.000  | 1.000    |
| ARTHROTEC.2   | 0.750     | 0.800  | 0.774    |
| ARTHROTEC.3   | 0.545     | 0.375  | 0.444    |
| ARTHROTEC.4   | 0.222     | 0.167  | 0.190    |
| ARTHROTEC.5   | 0.000     | 0.000  | 0.000    |

**ADR-Specific Performance (vs. `meddra` folder)**

| File          | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| ARTHROTEC.1   | 1.000     | 1.000  | 1.000    |
| ARTHROTEC.2   | 0.714     | 0.909  | 0.800    |
| ARTHROTEC.3   | 0.800     | 0.308  | 0.444    |
| ARTHROTEC.4   | 0.000     | 0.000  | 0.000    |

***

## 🛠️ Technologies Used

* **Language**: Python 3
* **Core Libraries**:
    * `pandas`: For data manipulation and analysis.
    * `google-generativeai`: For interacting with the Gemini API for NER.
    * `fuzzywuzzy`: For approximate string matching.
    * `sentence-transformers`: For generating text embeddings.
    * `scikit-learn`: For calculating cosine similarity.
* **Environment**: Jupyter Notebook / Google Colab

***

## 📂 File Structure

* `Assingment_1_Aryan.ipynb`: Performs initial data loading and analysis of entity types.
* `Assingment_2_and_5_Aryan.ipynb`: Runs the Named Entity Recognition task using the Gemini API and saves the results.
* `Assingment_3_and_5_Aryan.ipynb`: Evaluates the NER model's performance against the `original` ground truth.
* `Assingment_4_Aryan.ipynb`: Evaluates the NER model's performance for ADRs against the `meddra` ground truth.
* `Assingment_6_Aryan.ipynb`: Implements and compares the two entity linking approaches (string matching vs. embeddings).
