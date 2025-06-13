# CIS580 HW2 Coding Part

## Dependencies

We recommend you use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or other virtual environment managers (venv, virtualenv, etc.).
Here is an example of how to use `conda` to 
create and enter a new virtual environment.
```
conda create -n "cis580" python=3.10
conda activate cis580
```

As for the Python versions, anything **between Python 3.8 and 3.11** should be fine.


Before running the program, you need to install the dependencies using the following command.
```
pip install -r requirements.txt
```


## Usage

To run this program:

```
cd code
python main.py
```

We also provided some helper flags. Please check `main.py` for details. You can generate your visualizations with either PnP or P3P algorithm, but you still need to implement both of them. Although you are not asked to implement the renderer, you are still encouraged to look through the code of renderer as you may need to render your results by yourself next time. 

PS: remember to complete the `est_homography.py` with the function you just wrote for HW1.


## Debugging

It's recommended to run the program with `--debug` when you start to work on this homework since the rendering takes about 2 mins to finish on a PC. 

```
python main.py --debug
```

Also, note that the main program has several other args you can set, please have a look at line 40 in the `main.py` for more details: You can pass `--solver PnP` or `--solver P3P` to the program to toggle between the solving methods.

Note, we also provide the `.vscode` launch configuration for you to easily debug in VSCode.


## Customization

You can pass the `--click_point` to `main.py` to render the bottle and the drill at different places. 
