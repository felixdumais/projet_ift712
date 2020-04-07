# projet_ift712
projet_ift712 is the semester project for the course IFT712 at the science faculty of UniversitÃ© de Sherbrooke. IFT712 is a course given by the computer science department. The goal of this project was to develop a code that was able to use 6 different machine learning techniques to recognize pathologies from several chest X-ray images from the kaggle dataset [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/nih-chest-xrays/sample). The dataset is composed of 5606 images with a resolution of 1024 x 1024 px. Images are stored in PNG format.
## Getting Started
To download the source code on your desktop, launch `git clone https://github.com/felixdumais/projet_ift712.git` in a Linux terminal. Once everything is downloaded, all the source code should be in a folder called *projet_ift712*. 
Once the project is downloaded, create a new virtual environment. In the terminal write `virtualenv ~/.venv_nih_ift712`. Activate your new virtual environment with `source ~/.venv_nih_ift712/bin/activate`. Make sure you have the newest version of `pip` with the command `pip install --upgrade pip`.
Once everything is up to date, install all the packages listed in the file *requirements.txt* in your new virtual environment with the command `pip install -r requirements.txt`. 
Make sure that the kaggle API is correctly installed and that the file kaggle.json, containing your kaggle credentials, is saved in the right folder. To see how to install the kaggle API correctly, go to the section *Usage of Kaggle API* of the README.md. To see if the kaggle API is correctly installed, go in the section *Running the tests* of the README.md.
You can now parse arguments in the terminal to launch the file *src/main_ift712.py*. To better understand the arguments, launch `python src/main_ift712.py -h`
## Built With
* [Kaggle API](https://github.com/Kaggle/kaggle-api)
## Usage of Kaggle API
This project uses Kaggle API to download the dataset. Ensure you have Python 3 and the package manager `pip` installed.
### Installation
If the API is not already installed, run the following command to access the Kaggle API using the command line:
`pip install kaggle` (You may need to do `pip install --user kaggle` on Mac/Linux.  This is recommended if problems come up during the installation process.) Installations done through the root user (i.e. `sudo pip install kaggle`) will not work correctly unless you understand what you're doing.  Even then, they still might not work.  User installs are strongly recommended in the case of permissions errors.
If you run into a `kaggle: command not found` error, ensure that your python binaries are on your path.  You can see where `kaggle` is installed by doing `pip uninstall kaggle` and seeing where the binary is.  For a local user install on Linux, the default location is `~/.local/bin`.  On Windows, the default location is `$PYTHON_HOME/Scripts`.
### API Credentials
To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with `echo %HOMEPATH%`). You can define a shell environment variable `KAGGLE_CONFIG_DIR` to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json` (on Windows it will be `%KAGGLE_CONFIG_DIR%\kaggle.json`).
For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command: 
`chmod 600 ~/.kaggle/kaggle.json`
## Running the tests
To verify that the kaggle API, run the file test/kaggle_API_test.py in your terminal with the command `python test/kaggle_API_test.py`. If the API does not found the file kaggle.json this error appears *OSError: Could not find kaggle.json. Make sure it's located in ~/.kaggle/kaggle.json. Or use the environment method*. Go to the section *Usage of Kaggle API* to see where to install kaggle.json file.
## Authors
* **FÃ©lix Dumais (14053686)** 
* **Nicolas Fontaine (15203272)** 
* **Joelle FrÃ©chette-Viens (15057894)** 