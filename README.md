# Multi-Agent Reinforcement Learning

The aim of this project is to explore Reinforcement Learning approaches for Multi-Agent System problems. Multi-Agent Systems pose some key challenges which not present in Single Agent problems. These challenges can be grouped into 4 categories ([Reference](https://arxiv.org/abs/1810.05587)):
* Emergent Behavior
* Learning Communication
* Learning Cooperation
* Agent Modelling

We focus on the problem of learning communication and cooperation in multi agent systems. 

We also have a [blog](https://marl-ieee-nitk.github.io) with articles on the several concepts involved in the project.  

## Implementations
### Differentiable Inter Agent Learning
Run and experiment with the implementation in your browser:[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/MJ10/2c0d1972f3dd1edcc3cd17c636aac8d2/dial.ipynb)

[Foerster et al., 2016](https://arxiv.org/abs/1605.06676)

This is one of the seminal works in applying Deep Reinforcement Learning for learning communication in cooperative multi-agent environments. The paper proposes two learning approaches, Reinforced Inter Agent Learning (RIAL) and Differentiable Inter Agent Learning (DIAL). We implement the DIAL approach on the Switch Riddle environment.

The implementation in this repo is structured as follows:
* [`env/switch_riddle.py`](https://github.com/IEEE-NITK/Multi-Agent-Reinforcement-Learning/blob/master/env/switch_riddle.py): Contains the implementation of the Switch Riddle environment.
* [`agent.py`](https://github.com/IEEE-NITK/Multi-Agent-Reinforcement-Learning/blob/master/agent.py): Contains the implementation of the CNet model, Discretize/Regularise Unit and the Agent itself.
* [`arena`](https://github.com/IEEE-NITK/Multi-Agent-Reinforcement-Learning/blob/master/arena.py): Contains the code for training the algorithm on the environment. 

## Requirements
* [PyTorch](https://pytorch.org/)

## Team
* Moksh Jain
* Mahir Jain
* Madhuparna Bhowmik
* Akash Nair

Mentor: Ambareesh Prakash

## License
This repository is licensed under the [MIT License](https://github.com/IEEE-NITK/Multi-Agent-Reinforcement-Learning/blob/master/LICENSE.md). 