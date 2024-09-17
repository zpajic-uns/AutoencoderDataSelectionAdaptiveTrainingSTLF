# AutoencoderDataSelectionAdaptiveTrainingSTLF
 Autoencoder Data Selection Adaptive Training for Improved Performance and Accuracy of ANN Short-Term Load Forecast in ADMS

This repository contains all the data and code to reproduce the research results. Raw input data can be found in:
SimilarDayAutoencoderNNForecast\InputFiles

The results and trained models can be found in main folder:
SimilarDayAutoencoderNNForecast\SavedModel

For every date, there is a separate subfolder with results for different cases from the research. User can reproduce the evaluation if copy the models from the desired subfolder and place them in the main folder, and run the partucular script.

For setting up the virtual environment, please use Requirements.txt in your IDE or with pip install.

## Configuration of the scripts
Almost all needed parameters are configured in the Constants.py file. In this chapter, the main configuration parameters are described.

DAY_TO_PREDICT - Used to set the date that will be analyzed and predicted. Used in all scripts to identify day of interest. Example of usage:
DAY_TO_PREDICT = '2023-08-22'

AUTOENCODER_INPUT_ATTRIBUTES - Used to define input data that will be used in the autoencoder program. Currently available are "MinusOneDay_Load" and "Temp". Example of usage:
AUTOENCODER_INPUT_ATTRIBUTES = ['MinusOneDay_Load', 'Temp']

AUTOENCODER_CODE_DIMENSION - Used to define the dimension of the latent space of the autoencoder. Example of usage:
AUTOENCODER_CODE_DIMENSION = 40

NN_INPUT_DIM - Neural network input dimension. Example of usage:
NN_INPUT_DIM = 24

SELECT_SORTED_DAYS - True if algorithm should train the ANN only with selected days. If False, then the algorithm will use all available days. Example of usage:
SELECT_SORTED_DAYS = True

LOAD_SAVED_MODEL - If the pretrained model is used, the value should be set to True. Example of usage:
LOAD_SAVED_MODEL = False

PRETRAINED_MODEL_NAME - This string defines which model from the SavedModel folder should be loaded as pretrained model. Example of usage:
PRETRAINED_MODEL_NAME = 'model.h5'

USE_DISTANCES - True if the algorithm should use distances for selecting similar days. Example of usage:
USE_DISTANCES = True

USE_WEIGHTS - True if the algorithm should use weights in the training procedure. Example of usage:
USE_WEIGHTS = True

IS_FINE_TUNING - True if the training stage is the fine-tuning stage. Example of usage:
IS_FINE_TUNING = False

IS_ENSEMBLE_PRETRAINING - Used only for pretraining of the ensemble training. Example of usage:
IS_ENSEMBLE_PRETRAINING = True

NMB_SELECTED_DAYS - Defines how many days should algorithm use in the training dataset. Example of usage:
NMB_SELECTED_DAYS = 4

NMB_EPOCHS - Defines the number of epochs in the training procedure. Example of usage:
NMB_EPOCHS = 5000

MIN_LR - Minimal learning rate that is used in the Reduce Learning Rate procedure during training procedure. Example of usage:
MIN_LR = 0.0001

## Program Sequence
The whole sequence of the algorithm:
1. Run PreprocessDayData.py
2. Run AutoencoderClass.py
3. Run CodeDays.py
4. Run SelectCodedDays.py
5. Run MakeAndTrainNN.py
6. Set the model name for evaluation in the script and run EvaluateNNResults.py
7. Run TrainEnsemble.py
8. Run EvaluateEnsemble.py

If DAY_TO_PREDICT is changed, it is enough to run the first four steps once. There is no need to run these four steps furthermore because the intermediate data is stored in files which are used in the rest of the scripts.

## Using existing trained models to reproduce results
Place the desired model in the SavedModel directory, copy the name of the model, and change PRETRAINED_MODEL_NAME parameter. Run the scripts for evaluation.

The EvaluateEnsemble.py script requires four models named 0_model.h5 to 3_model.h5. This can be changed manually if the names are different.
