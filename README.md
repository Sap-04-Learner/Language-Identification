# Language-Identification

## Project Overview
> The goal of this project is to create a deep learning model that can detect the language of spoken words by analyzing audio files. The dataset consists of audio files from three different languages: English, Spanish, and German. The model processes the audio data and predicts the language based on the features extracted from the audio signals.

## Dataset
The dataset used in this project consists of `.flac` audio files. The dataset is divided into a training set and a test set. The files are named with the following convention:

- Files starting with `en` are in English
- Files starting with `es` are in Spanish
- Files starting with `de` are in German

The files are organized into separate train and test folders, with the first 1800 files being in English, the next 1800 in Spanish, and the last 1800 in German.

**To access the dataset, please click on the following link: [Language_Dataset](https://drive.google.com/drive/folders/1_xQUO-yui_V8RbEm6Ugz6q5AD0rH0Mah?usp=sharing)**

>  To access the dataset from where this dataset is derived, click on these links:
>  * **GitHub Repository:** [GitHub Link](https://github.com/tomasz-oponowicz/spoken_language_dataset/tree/master)  
*or*
>  * **Kaggle Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/toponowicz/spoken-language-identification/data?select=train)



### Directory Structure:
```
/dataset
 |__  /train
 |     |__ en_001.flac
 |     |__ en_002.flac
 |     |__ ...
 |     |__ es_1801.flac
 |     |__ es_1802.flac
 |     |__ ...
 |     |__ de_3601.flac
 |     |__ de_3602.flac
 |     |__ ...
 |__ /test
      |__ en_3601.flac
      |__ ...
      |__ es_5401.flac
      |__ ...
      |__ de_7201.flac
      |__ ...

```

## Dependencies
The following dependencies are required to run this project:

- Python 3.7+
- TensorFlow 2.x
- librosa
- numpy
- matplotlib
- scikit-learn
- ipython
- seaborn

You can install the required libraries using pip:
```
pip install numpy librosa scikit-learn tensorflow matplotlib seaborn
```

> *If you're using Google Colab, you don't need to install these libraries manually. They are already pre-installed in the Colab environment.*

## Getting Started
### Setup
Clone the repository:

- Option 1: Google Colab
    - Create a new notebook on Google Colab.
    - Upload the [`speech_recognition.ipynb`](https://colab.research.google.com/drive/1b2iAc8ye8DPcHz3LIHYa2N2o1oqQZgXs?usp=drive_link) notebook to your Colab workspace.
    
- Option 2: Local Setup
    - Clone the repository to your local machine.
    - Set up a Python environment with the required libraries from the dependencies.

> *As most local systems lack the GPU power required for heavy computational tasks, it is recommended to use Google Colab for running this project.*

### Training the Model
To train the model, you can use Google Colab. Follow these steps:

1. Open the provided Colab notebook (or your own Colab notebook if applicable).

2. Upload the dataset to your Colab environment. You can either upload the dataset directly to Colab or use Google Drive to store it. If you are using Google Drive, mount it in Colab by running the following code:

```
from google.colab import drive
drive.mount('/content/drive')
```
3. Probably the path used in code to access the dataset will be same if you are downloading the dataset [Language_Dataset](https://drive.google.com/drive/folders/1_xQUO-yui_V8RbEm6Ugz6q5AD0rH0Mah?usp=sharing) in your Google Drive.

4. Follow the instructions as mentioned in the [`speech_recognition.ipynb`](https://colab.research.google.com/drive/1b2iAc8ye8DPcHz3LIHYa2N2o1oqQZgXs?usp=drive_link) to train the models and testing it against test cases.

> - **Accessing the Trained Models**
> : The trained models required for inference are available on Google Drive.
> - **Download Link:** [Language_Detection](https://drive.google.com/drive/folders/1gmQlrOGMAMqTdIFNIXgxvxfO7VCmEvt8?usp=drive_link)


**Recommended Approach:**

1. **Download the Pre-trained Models:** Download the provided models from the Google Drive link.
2. **Upload to Google Drive:** Upload the downloaded models to your Google Drive for easy access in your Colab notebook.

**Alternative Approach:**

1. **Train Your Own Model:** Run the training notebook to train your own models.
2. **Use the Trained Model:** Use the newly trained models for inference.

## Inference
**Using the Pre-trained Models for Inference:** 

You can refer to this Google Collab Notebook: [speech_recognition_inference.ipynb](https://colab.research.google.com/drive/1h4NAWyvdK6pO7YmH8rPwypeTsGFBtJTm?usp=drive_link)

1. **Open the Inference Notebook:** Open the provided inference notebook in Google Colab.
2. **Upload the Trained Models:** Upload the downloaded models from Google Drive to your Colab notebook's runtime environment.
3. **Set the Model Path:** Adjust the model path in the notebook to point to the uploaded models.
4. **Run the Inference Code:** Execute the inference code to process your audio input and obtain language predictions.

**Note:** Ensure that the audio input format and preprocessing steps match the requirements of the trained models.

> ### Optional
> **Use Your Own Audio File for Prediction To** predict the language for your own `.flac` audio file, upload it to the Colab environment and follow the above steps. Ensure that the audio is in `.flac` format for accurate predictions. Just refer to your audio file path in `AUDIO_FILES`.
