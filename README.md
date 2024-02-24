# identification-of-medicinal-plants

This Streamlit application allows users to classify images of medicinal plants using a pre-trained deep learning model. The model is capable of predicting the species of the plant based on the uploaded image
## Getting Started

To run this application locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running:
pip install -r requirements.txt (or) you can use the following commands:
pip install tensorflow
pip install  streamlit 
 
4. Run the Streamlit app by executing:
streamlit run frontend.py

## Usage

1. Upon running the app, you will see a file uploader where you can choose an image of a medicinal plant.
2. After uploading an image, the app will display the uploaded image and classify it using the pre-trained model.
3. The predicted class along with information about the medicinal uses of the plant (if available) will be shown.

## Data Source
The dataset used to train the model is mendley datset which contains 6o different indian plant species.

## Model Details

The deep learning model used in this application is an Xception model trained on a dataset of medicinal plants. The model achieves high accuracy in classifying various species of plants.





