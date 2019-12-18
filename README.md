# Refinement-on-AlexNet-for-TimeSeries
This is not the creation of my own. This repository is just improving the code from https://github.com/hfawaz/InceptionTime.

The main difference from the original source is that the user can tune all the parameters in the config.json, which make the program more user-friendly.
Also, more explanation comments are added in the code.

##### Remarks
For the details of inception model, please refer to the original README.MD and read the paper mentioned in it.


## How To Use

### Run InceptionTime on the Archive
You should issue the following command ```python3 main.py InceptionTime```. 

### Run the hyperparameter test for InceptionTime on the Archive
You should issue the following command ```python3 main.py InceptionTime_xp```. 

### Create an new dataset based on a sample dataset
You should first issue the following command ```python3 main.py run_length_xps``` to generate the resamples.

### Generate result of trained InceptionTime model
You should first issue the following command ```python3 main.py generate_results_csv```.


## Configuration File
The configuration file is 'config.json'. The following section is to explain different parts in the config.

### PATH
The 'root_directory' is the directory of the source code.

### ARCHIVE
The 'archive_name' has to indicate the name of folder which contains all the datasets.

The 'data_name_list' has to list all the datasets to be trained by InceptionTime.

### INCEPTION_MODEL
This part is to tune the hyperparameter for the InceptionTime model.

##### Train
The 'num_of_Inception_modules' is the number of inception layers (or modules) in the Neural Network.

The 'default_params' is default parameters of InceptionTime model to be trained.
* 'verbose' means whether to print the summary of InceptionTime model in console
* 'build' means whether to initiate the neural network for the InceptionTime model
* 'batch_size' means the batch size for each training batch
* 'nb_filters' means the filters for the convolution layer in inception layers (or modules)
* 'use_residual' create shortcut layer in the inception layers (or modules)
* 'use_bottleneck' means whether to use bottleneck in inception layers (or modules)
* 'depth' means the depth inside the inception layers (or modules)
* 'kernel_size' means the kernel size of convolution part in the inception layers (or modules)
* 'nb_epochs' means the epochs (times) will be trained on whole archive

The details in the above 'default_params' can be found in the paper.

##### HyperParameter_test
The part is to indicate the a set of hyperparameters to be test. This section is to find the best combination of the hyperparameters for the InceptionTime on the indicated archive.
The result of ```python3 main.py InceptionTime_xp``` would be affected by this section.

### DATA_AUGMENTATION
This part is to generate resamples, which would affect the result of ```python3 main.py run_length_xps```.
The resamples are based on the indicated 'sample_dataset'.


