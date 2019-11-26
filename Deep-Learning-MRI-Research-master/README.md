# Deep-Learning-MRI-Research

Culmination of the deep learning research conducted throughout my MS. This should be fairly easy to get up and running, but if there are any issues, don't hesitate to reach out.


## Instructions ##

* Run "./configure.sh" in a terminal to install all dependencies and pull down BRATS 2018 data.
* Run "Driver.py" either in a terminal or in your favorite IDE - it should create a U-Net, load in some of the BRATS data, and start training
* It's configured to use GPU resources if their available, or fall back to CPU otherwise
* To change the data that's being loaded (i.e; adding your own dataset), edit the list of data directories in "TrainModel.py"
* To swap out the U-Net for an Inception U-Net, just change "createUNet" to "createInceptionUNet" in "TrainModel.py"
* Currently doing a 90/10 training/validation split - that can be changed by modifying the "validation_split" parameter in "TrainModel.py"
* Currently using all modalities, and predicting all three segments (edema/non-enhancing/enhancing) - to change this, just edit the list in "TrainModel.py"
* Testing/generating visualizations can be done with "TestModel.py" (although confirm the configurations match the settings used during training)
