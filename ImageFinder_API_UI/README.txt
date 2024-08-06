# Image Similarity Retrieval System : ImageFinder

## Repository Structure
This repository contains an image similarity retrieval system implemented using PyTorch and FastAPI. The main components are:

Image_findder.py: The core script containing the image processing logic 
api_APP.py : FastAPI server
UI_StreamLit.py : Frontend UI for API with Streamlit
download_image.py: A python script to download the files from the above csv
list_image.csv : .csv for images
/images: Directory containing the dataset of images for similarity comparison


## How to Use

1. Ensure your image dataset is in the /images directory built with download_image.py.
2. Run the server: api_APP, the API will be available at 127.0.0.1:8000/docs
3. For API Rest only, go to the /find_similar endpoint with a POST request, uploading an image file
4. For the UI API with Streamlit, run UI_StreamLit.py, then copy/paste the line, for me it's "streamlit run d:\DOWNLOADS\Lydia_APP\UI_StreamLit.py [ARGUMENTS]"
5. Enjoy


## Technical Choices

**Deep Learning Framework**: PyTorch 
**Pre-trained Model**: I used Resnet50 first, but due to image dataset with higher resolution images, EfficientNet is better for this. The last fully connected layer is removed to use it as a feature extractor.
**Feature Extraction**: The images are processed through the EfficientNet model before flattening, 
**Similarity Metric**: Cosine similarity is used to compare image features. This metric is effective for high-dimensional data and is less affected by the magnitude of the features, focusing instead on the direction of the vectors in the feature space.
**API Framework**: FastAPI with easy-to-use syntax, and automatic API documentation generation. It's faster to use than flask in my case.
**Image Preprocessing**: A standard image transformation pipeline is used, including resizing, center cropping, conversion to tensor, and normalization. This ensures consistency in input size and format for the neural network.
**UI**: Frontend was not requested but it looks better for a app.
