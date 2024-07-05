Repository Overview
This repository contains code and resources for building and tuning classification models, specifically focusing on decision trees.

Modules
1. Classification_model
This module contains various decision tree models and their hyperparameters, which can be customized for optimal performance.

2. cut_tuning
This module includes code for designing and fine tuning a linear cut between log likelihood ratios/softmax values and other reconstructed variables. A quadratic cut line is also defined but not implemented as it requires too much time.

3. Create_file
This module provides code to select the relevant variables required for analysis and to create a pandas DataFrame from the raw data.

4. main_model
This Python script is used to run the BDT model. Various parameters can be set within the code to tailor the model to specific requirements.

5. main_tuner
This Python script is designed to run the cut tuning algorithm. Similar to main_model, different parameters can be adjusted within the code to optimize the tuning process.
