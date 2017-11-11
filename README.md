# Lunar Lander v2

## Problem Statement
[Lunar Lander v2](https://gym.openai.com/envs/LunarLander-v2/) is an package in Open AI's gym. The aim is to learn how to land a space shuttle correctly on a launch pad without crashing. A common approach to solve such problem is to use Q Learning but it works only in a discreet space, discreetizing the whole space would lead to a large number of states to keep track of. Instead for this project we used a neural network to map the continuous input state to the Q values for state action pair required for Q learning to work.

* Input State: **R**<sup>8</sup>
* Possible Actions: 4  {do nothing, fire left orientation engine, fire main engine, fire right orientation engine }

## Network Architecture

Following is the network used 

![Network Architecture](https://github.com/monkeydunkey/Lunar_Lander_v2/blob/master/resources/neuralNetworkArchitecture.png)

## Tips and Tricks to make the network learn
1. Use a memory (a simple list) to store randomly sample onservations of the form <S, A, S', R> where S is the State, A the action taken in that state, S' the next state and R the reward returned.

2. Try multiple epsilon greedy strategies to balance between exploration and exploitation. The general strategy should be such that the bot does a lot of exploration in the beginning by performing random actions and as the network learns it should use it more and more to better fine tune the knowledge it has about the problem. In our case we employed a linear decay till some 100 episodes followed by exponential decay

3. Ensure that the batch size is large enough for the network to generalize, if the batch is too small the loss will just thrash around

4. Use another network with the same architecture to get the next state Q values. This network should not be trained but rather the values from the main network should be copied to it at regular intervals. The reason for using this is that using the same network that is being trained to get Q values while training, leads to unstability in the model

5. Use Huber loss instead of Mean Square Error as it seems to be more stable.

6. Use Tensorboard to visualize the gradient flow and the loss, this really helps in understanding and answering the question of whether or not the model is learning anything
