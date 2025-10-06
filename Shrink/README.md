# SHRINK: Data Compression by Semantic Extraction and Residuals Encoding
Welcome to the repository accompanying our IEEE BIG DATA 2024 paper [SHRINK: Data Compression by Semantic Extraction and Residuals Encoding
](https://arxiv.org/abs/2410.06713). We propose a new type of compression algorithm called SHRNK that produces a high efficient encoding for time series data using PLA technique.


## Dataset files
Download from the following link in SIMPIECE paper:

- https://github.com/xkitsios/sim-piece
- Location:
src/test/resources

## Requirements:
- Create a new Python virtual environment 
- Install the dependencies via `pip install -r requirements.txt`
- We used TRC as our downstream compression as explained in the paper. It can be installed through the link: https://github.com/powturbo/Turbo-Range-Coder
- Compile for TRC:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Download or clone TurboRC<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;git clone --recursive https://github.com/powturbo/Turbo-Range-Coder.git<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cd Turbo-Range-Coder<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;make<br>


## Our method: SHRINK
- Located in `shrink.py`
- `Hyperparameter` setting mannually hyperparameter self.alpha for  L

Detailed comments explaining the correspondence beteween parts of the code and the paper can be found throughout.  
## Main experiment file
1) Navigate to `Experiments.py` and choose errorthreshold you would like to run in `epsilons` and Base errorthrolds in `BaseEpsilons`
2) download datasets and Change data path accordingly
3) Run file


# Citation
If you use SHRINK in your paper, please use the following citation:

`@inproceedings{sun2024shrink,
  title={SHRINK: Data Compression by Semantic Extraction and Residuals Encoding},
  author={Sun, Guoyou and Karras, Panagiotis and Zhang, Qi},
  booktitle={2024 IEEE International Conference on Big Data (BigData)},
  pages={650--659},
  year={2024},
  organization={IEEE}
}`

