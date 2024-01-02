## How to Deploy:

1. Open up a terminal (i.e., Windows Command Prompt) and navigate to the directory "Flask App".

2. Check to make sure you have Python and pip installed. Check if Python is installed by typing `python --version`. If Python is not installed, it can be downloaded for Windows through the following link [Python Downloads](https://www.python.org/downloads/windows/). Click on either "Download Windows installer (32-bit)" or "Download Windows installer (64-bit)" depending on your system. Once Python is downloaded, check in your terminal whether pip was installed as well by typing `pip -V`. If pip is not installed, run the command:
   
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

Then type this command to install pip:
`python get-pip.py`


3. Next the jupyter package will need to be installed with `pip install jupyter`. Then type the command `python -m notebook` which 
should open a web browser which shows the "Flask App" directory in the jupyter notebook home page. Open the jupyter notebook "install.ipynb".
Once opened click run and the Python libraries should start installing

4. Once the packages, stop the juypter notebook server in terminal by clicking control-c. Next you have to run the main Python file to set up the Flask server. So type the command `python main.py`. Some text will be displayed as a result, the last line should be
Running on http://127.0.0.1:5000/ (Press CTRL+C to quit). Copy http://127.0.0.1:5000/ and paste it in your browser, click enter and the website should be displayed



   
