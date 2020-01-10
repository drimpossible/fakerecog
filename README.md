# fakerecog

Code for my submission to the deepfake detection challenge. Currently, this is not very helpful. Please overlook this. To be updated once we have a first cut code ready.

# Creating dataset files


=======
# Deepfake Recognition

This repository contains the code for our submission to the deepfake detection challenge.

## Installation and Dependencies

Install all requirements required to run the code by:
	
	# Activate a new virtual environment
	$ pip install -r requirements.txt

## Usage

* The current directory structure of the project is as follows:
 ``` 
  $  fakerecog/
	$    data/
	$    face-detect/
	$    deepfake-recog/
	$	     models/
  $      datasets/
```
* [Download](https://www.kaggle.com/c/deepfake-detection-challenge/data) the deepfake challenge dataset and the deepfake preview dataset. Then, extract the zip files and ready the dataset for preprocessing like face detection.
  
   	$ python test_datasets.py

* Run the code by the following command in face detection:

      $ python main.py 
  
Additional hyperparameters can be found in `opts.py` file.

