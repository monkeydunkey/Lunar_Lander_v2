# Lunar Lander v2

## Problem Statement
[Lunar Lander v2](https://gym.openai.com/envs/LunarLander-v2/) is an package in Open AI's gym. The aim is to learn how to land a space shuttle correctly on a launch pad without crashing. A common approach to solve such problem is to use Q Learning but it works only in a discreet space, discreetizing the whole space would lead to a large number of states to keep track of. Instead for this project we used a neural network to map the continuous input state to the Q values for state action pair required for Q learning to work.

Input State: **R**<sup>8</sup>
Possible Actions: 4  {do nothing, fire left orientation engine, fire main engine, fire right orientation engine }

## Network Architecture

Following is the network used 
