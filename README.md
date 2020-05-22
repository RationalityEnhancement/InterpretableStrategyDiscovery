# AI-Interpret

This is a Python library associated with the "Automatic Discovery of Interpretable Planning Strategies" (Skirzynski, Becker, & Lieder, 2020) submitted to Special Issue on Reinforcement Learning for Real Life in Machine Learning Jornal. Manuscript is avaiable at .... AI-Interpret is an imitation learning algorithm that finds the least complex, high-performing MAP logical formula that interprets a set of demonstrations.

We used Python 3.6.9 and Ubuntu 18.04.

## Installation

To install the library, simply clone the repository and use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
git clone https://github.com/RationalityEnhancement/interpretable-strategy-discovery.git
cd RL2DT
pip install -r requirements.txt
```
## Data
Download the demonstrations and other data important for simulations as the [PLP_data](https://owncloud.tuebingen.mpg.de/index.php/s/jM8SfdJxsgXdLWb) folder. Extract the folder to the ```RL2DT``` directory.

## Example Usage

To run the whole pipeline and interpret custom demonstrations (read more about the parameters in ```interpret.py```):
```bash
python3 interpret.py --algorithm adaptive --elbow_choice automatic --candidate_clusters 3 4 5 6 7 8 9 10 12 12 13 14 15 16 17 18 19 20 30 40 50 100 200 300 --num_candidates 4 --custom_data True --demo_path ./PLP_data/copycat_64_constant_paper.pkl --mean_reward 9.33 --num_demos 64 --interpret_size 5 --num_rollouts 100000 --aspiration_level .7 --tolerance 0.02 --info TRIAL
```

To only generate a plot with clustering values for different numbers of clusters (which is one of the steps done by the previous call):
```bash
python3 elbow_analysis.py --num_demos 8 --demo_path ./PLP_data/copycat_64_different.pkl --candidate_clusters 3 4 5 6 7 8 9 10 12 12 13 14 15 16 17 18 19 20 30 40 50 100 200 300 --num_candidates 4 --info TRIAL2
```

Same for the files which are currently in the ```PLP_data``` folder (with default values for the parameters):

```bash
python3 interpret.py --algorithm adaptive --environment constant_paper --num_demos 64 --interpret_size 5 --aspiration_level .7 --name_dsl_data DSL_64_constant_paper --info TRIAL_MDP
python3 elbow_analysis.py --num_demos 8 --name_dsl_data DSL_8_different --info TRIAL_MDP2
```

To visualize the learned formulas as decision trees:
```bash
python3 formula_visualization --data ./interprets/adaptive_demos_64_constant_paper_aspiration_0.75_depth_4_validation_cluster_0.3_num_clusters_18.pkl --only_dot False
```
