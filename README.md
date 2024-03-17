# EIGN: Equivariant Interaction-aware Graph Network for Predicting the Binding Affinity of Protein-Ligand
a method for predicting the binding affinity of protein-Ligand
# Installing the eign Virtual Environment on Linux

This document will guide you through the process of installing and configuring the `eign` Anaconda virtual environment on a Linux system. This environment contains various popular machine learning, scientific computing, and data processing libraries, such as TensorFlow, PyTorch, Scikit-learn, Pandas, and more.

## Prerequisites

Before you begin, ensure that you have Anaconda Distribution installed on your Linux system. If you haven't installed it yet, please visit the [Anaconda website](https://www.anaconda.com/distribution/) and follow the instructions to download and install Anaconda.

## Steps

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

That's it! You have now successfully installed and configured the `eign` virtual environment on your Linux system. Whenever you need to use this environment, make sure to activate it first by running `conda activate eign`. Then, you can run your code or scripts within the activated environment.

If you need to install additional dependencies or update existing ones, you can use the `conda install` or `pip install` commands while the `eign` environment is activated.


