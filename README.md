# Puck_Catching_Agent

This repository contains several policy gradient implementations (as well as RQN) of Puck_Catching_Agent designed in [physics_learning_rl][physics_learning_rl]. For different algorithms' experiment, evaluation and optimization.

Using the same simulator as used in [physics_learning_rl][physics_learning_rl].

  [physics_learning_rl]: https://github.com/mingxuanM/physics_learning_rl

## Implemented

1. RQN (Recurrent Q-Network) with LSTM

   This is the pytorch version of the RQN Puck_Catching_Agent (the pretrined-model in transfer learning framework) used in [physics_learning_rl][physics_learning_rl].

2. Actor-Critics
   
   Update A-C networks during each step, both with LSTM. Critics predicts action value.

## Algorithms plan to have

1. Advantage Actor-Critics (A2C)
   
   Update A-C networks after a whole episode, both with LSTM. Critics predicts state value.

## Further task
Multi-agent framework: 
1. Let agents learn to catch a same puck. 
2. team-work to control other pucks to push a target puck to goal.

## Package dependency
* python 3.7+
* pytorch 1.2.0+
* pyduktape 0.0.6
