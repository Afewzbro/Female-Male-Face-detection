# Female-Male-Face-detection

This project implements a binary image classifier to distinguish between female and male images. 
It uses a Convolutional Neural Network (CNN) architecture trained on a dataset of female and male images of different races.

## Dataset

The dataset used in this project consists of female and male face images. 
The dataset is organized into three directories: train, validation, and test. 
Each directory contains separate subdirectories for females and males.
The [DATASET](https://drive.google.com/drive/folders/1TkGfbKsMEL380bDS1VGmhBGc95xNr7bq?usp=sharing) I used.
The [DATASET](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset) I downloaded from Kaggle.

# Dependencies

The following libraries are required to run the code:

- TensorFlow
- NumPy
- Matplotlib
- PIL
- Pandas
- scikit-learn

You can install the required dependencies by running the following command: pip install tensorflow numpy matplotlib pillow pandas scikit-learn


## Usage

1. Clone the repository: [The Code](https://github.com/Afewzbro/Female-Male-Face-detection) link
   - git clone 

2. Preprocess the Dataset:
   - Place your dataset in the appropriate directories: `dataset/train`, `dataset/validation`, and `dataset/test`.
   - Modify the paths in the code to match the location of your dataset.

3. Train the Model:
   - Run the `cat_vs_dog_classifier.py` script to train the model.
   - Adjust the hyperparameters and model architecture as needed.

4. Evaluate the Model:
   - The trained model will be evaluated on the test dataset automatically.
   - The evaluation results, including test loss and accuracy, will be displayed.

5. Generate the Confusion Matrix:
   - The confusion matrix will be generated after evaluating the model.
   - The matrix will show the distribution of correct and incorrect predictions for each class.

## Results

- The model achieved a test accuracy of 97.55% and a test loss of 0.0956.
- The confusion matrix shows a balanced distribution of correct predictions for both cats and dogs.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and adapt this code for your own purposes. Any contributions are welcome!


