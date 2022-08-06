<p align="center">
  <img alt="📉FO_ _Prox" src="https://user-images.githubusercontent.com/62103572/183243536-c02b7744-6b40-462a-9c08-5c44b6e8d9fd.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/FO-PROX-first-order-and-proximal-methods-convergence-comparison?style=social">
</p>


Implementation of different first order (FO) methods and proximal gradient methods: Gradient descent (GD), Gradient descent strongly convex (GDstr), Accelerated gradient descent (AGD), Accelerated gradient descent strongly convex (AGDstr), Accelerated gradient descent with restart (AGDR), Adaptive Gradient Method (AdaGrad), Stochastic gradient descent (SGD), Stochastic averaging gradient (SAG), Stochastic gradient descent with variance reduction (SVR), Subgradient method (SubG), Iterative shrinkage-thresholding algorithm (ISTA), Fast iterative shrinkage-thresholding algorithm (and with restart) (FISTA and FISTAR), Stochastic proximal gradient method (prox_sg). 

For a more detailed explanation of the terms mentioned above, please read *Exercise instructions.pdf*, it contains also some theoretical questions answered in *Answers.pdf* (handwritten). 

The project was part of an assignment for the EPFL course [EE-556 Mathematics of data: from theory to computation](https://edu.epfl.ch/coursebook/en/mathematics-of-data-from-theory-to-computation-EE-556). The backbone of the code structure to run the experiments was already given by the professor and his assistants, what I had to do was to implement the core of the optimization steps, which are the FO and proximal methods algorithms and other minor components. Hence, every code file is a combination of my personal code and the code that was given us by the professor.

The following image shows an example of the output figures of the code **run_comparison.py**. All plots show a comparison of convergence speed for different optimization methods, the y-axis shows how far the model is from its optimum (the smaller, the less far, the better)

<p align="center">
<img width="500" alt="Immagine 2022-08-05 155002" src="https://user-images.githubusercontent.com/62103572/183244639-4fc62501-f0e9-468a-b641-b44e4d47883e.png">
<img width="500" alt="Immagine 2022-08-05 155002" src="https://user-images.githubusercontent.com/62103572/183244641-680a9f9b-a628-4720-9b27-adc77ed65966.png">
<img width="500" alt="Immagine 2022-08-05 155002" src="https://user-images.githubusercontent.com/62103572/183244643-11126344-87b5-473b-aa7c-aa6fbb51b745.png">
</p>


The following image shows an example of the output figures of the code **run_theoretical_vs_emprical_conv_rate.py**. It's a comparison between the theoretical upper bound of the optimization method and the actual empirical convergence rate.

<p align="center">
<img width="500" alt="Immagine 2022-08-05 155002" src="https://user-images.githubusercontent.com/62103572/183244791-7524be62-8955-48b6-9fae-94c4d11f818c.png">
<img width="500" alt="Immagine 2022-08-05 155002" src="https://user-images.githubusercontent.com/62103572/183244792-bd8fe301-b6cd-4814-9333-be2edc364bcc.png">
</p>


## Author
-  [Elia Fantini](https://github.com/EliaFantini)

## How to install and reproduce results
Download this repository as a zip file and extract it into a folder
The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/

Additional package required are: 
- matplotlib
- sklearn
- scipy

To install them write the following command on Anaconda Prompt (anaconda3):
```shell
cd *THE_FOLDER_PATH_WHERE_YOU_DOWNLOADED_AND_EXTRACTED_THIS_REPOSITORY*
```
Then write for each of the mentioned packages:
```shell
conda install *PACKAGE_NAME*
```
Some packages might require more complex installation procedures. If the above command doesn't work for a package, just google "How to install *PACKAGE_NAME* on *YOUR_MACHINE'S_OS*" and follow those guides.

Finally, run **run_comparison.py** and **run_theoretical_vs_emprical_conv_rate.py**. 
```shell
python run_comparison.py
python run_theoretical_vs_emprical_conv_rate.py
```

## Files description

- **code/figs/** : folder containing outplot plots produced by the code
- **code/dataset/**: folder containing the dataset to train and test on

- **code/core**: folder all the code to run the training and testing, included the optimization algorithms

- **run_comparison.py**: main code to run the training and testing

- **run_theoretical_vs_emprical_conv_rate.py**: main code to run the training and testing

- **Answers.pdf**: pdf with the answers and plots to the assignment of the course

- **Exercise instructions.py**: pdf with the questions of the assignment of the course

## 🛠 Skills
Python, Matplotlib. Machine learning, proximal methods, theoretical Analysis knowledge of convergence rates, implementation of optimization methods.

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
