# OpenAI-Gym-Taxi-v2

# Overview

This is a mini project wherein I use OpenAI Gym's Taxi-v2 environment to design an algorithm to teach a taxi agent to navigate a small gridworld.

There are three key files:
1. Main.py
2. Monitory.py
3. Agent.py

I used the Expected Sarsa (Temporal-Difference) algorithm when controlling agent's next steps.

The primary hyperparameters are the following (which can be found in the Agent.py file):
Epsilon - to define the epsilon-greedy policy to determine next actions
Gamma - to define the discount rate applied to future values
Alpha - to define the learning rate associated with updating the action value function (Q) at each time step

The current hyperparamaters yield a best average reward of ~9.0. 

Further tuning of the above hyperparameters is recommended to help drive a better average reward value.


