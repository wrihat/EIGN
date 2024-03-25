# EIGN: Equivariant Interaction-aware Graph Network for Predicting the Binding Affinity of Protein-Ligand
a method for predicting the binding affinity of protein-Ligand
# Installing the eign Virtual Environment

This repository is the implementation of EIGN. This document will describe the steps for setting up the environment, as well as the steps for testing our trained model.

## Installation


### Installation Steps

1. **Create the Virtual Environment**

   Open a terminal and run the following command to create a new virtual environment named `eign` with Python 3.8:

   ```bash
   conda create -n eign python=3.8
   ```

   This will create a new virtual environment with Python 3.8 installed.

2. **Activate the Virtual Environment**

   In the same terminal, activate the `eign` virtual environment by running:

   ```bash
   conda activate eign
   ```

   After activation, you should see the `(eign)` prefix in your terminal prompt.

3. **Install Dependencies**

   With the `eign` environment activated, use the following command to install all dependencies from the provided YAML file:

   ```bash
   conda env update -n eign -f /path/to/environment.yaml
   ```

   Replace `/path/to/environment.yaml` with the actual path to the YAML file on your system.

   This process may take some time, as it will download and install all required packages and dependencies. Be patient and let it complete.

4. **Verify Installation**

   After the installation is complete, you can try importing some of the installed libraries, such as TensorFlow and PyTorch, to verify that they are working correctly. For example:

   ```python
   import paddle
   import pgl
   ```
   If no errors occur, it means these libraries have been successfully installed.

## Test Trained EIGN

    The trained model is saved in ./trained directory, and the test dataset is placed under the ./dataset directory, please make sure the files in the response directory have been downloaded.Then  run the command at the command line:
    
   ```bash
   nohup python test.py --cuda 0 --batch_size 2 > ./running_logs/test_logs/pdbbind2016_EIGN.txt 2>&1 &
   ```


    






