# Epigenetic Biological Age

Model development for predicting and interpreting epigenetic biological age acceleration from wearable device data.

## Installation

This project requires Python 3.8 or later.

First, git clone the repository [git://github.com/epigenetic-biological-age]. Then, navigate to the project directory: `cd epigenetic-biological-age`.
Ensure that the pip version you are runnning is >= 20.0.2. 
We recommend using a virtual environment. Activate the virtual environment and install the project dependencies: `pip install -r requirements.txt`

## Usage

This project uses a `Makefile` for managing tasks. You can use the `make` command followed by the name of the task you want to run.
For example, if you want to train the model you can run the `train` task, by using the following command: `make train`. If you want
to run data through the existing trained model, then you would run either `make notrain`.

License