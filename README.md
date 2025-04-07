# Project_VideoGameAgent
Use the Gymnasium reinforcement learning environment to train an agent to play Prof. Tallman's side-scrolling game. You are welcome to use the original version of the game, as provided by Prof. Tallman, or change it to include your own modifications—such as new levels or gameplay adjustments. You may choose from several approaches to train your model:

Unregistered custom environment: Manually instantiate your environment object, similar to how we used Frozen Lake and Taxi, but without the gym.make() function.
Registered custom environment: Create a Gym-compatible environment that is formally registered, closely resembling the structure of Frozen Lake and Taxi.
Third-party training framework: Use an external library such as Stable-Baselines3 (note: this option has not been covered in class).
Reinforcement learning training can be time-consuming and may offer limited immediate feedback on the agent's actual performance. To evaluate your agent’s progress, you should periodically capture snapshots during training. The two easiest ways to do this are:

Saving the Q-table (or ANN weights) to a file every n episodes (choose an appropriate value for n).
Recording a video of the agent playing the game every n episodes.
Saving the Q-table/weights is recommended, as it allows you to pause and resume training later without losing progress. However, recording gameplay videos can be useful for visualizing your agent’s behavior over time. If you choose to record videos (or do both), consider using the gymnasium.wrappers.RecordVideo wrapper to automate this process.

This project falls right at the transition point between traditional Q-learning and Deep Q-learning with neural networks. You are free to choose either approach. Whichever method you use, you must submit a saved version of your trained agent that Prof. Tallman can render directly—without re-running the full training process.

For Q-learning, this usually means saving the Q-table (e.g., using pickle to write it to a file).
For Deep Q-learning, this typically involves saving your model’s parameters (e.g., with torch.save() in PyTorch or model.save() in TensorFlow).
Make sure your submission includes all the code necessary to load the saved agent and render a short demo of it playing the game.

Game Assets:

I don't expect anyone to change the game’s assets, but you’re welcome to add your own graphics or music if you’d like. If you use assets created by others, be sure their license permits such use.

Groups:

Feel free to collaborate with your friends for the basic setup of this project. You are welcome to discuss the game engine, how to create a custom environment, strategies for learning, and how to playback training snapshots. However, each person must design their own unique reward system so that our agents all have different behavior.

Grading:

This project isn't graded by a fixed rubric, but I believe that strong reinforcement learning projects tend to share several important characteristics.

An agent that plays the game with some level of success, which will largely depend on:
Creating an internally consistent custom environment that is suitable for training.
Well-designed observation states that provide the agent with informative values that can be efficiently processed and help the agent distinguish between different situations.
A thoughtful reward structure that balances short-term incentives with long-term success.
Demonstration of training progress, including the ability to render the agent's behavior at various stages of training. You must provide the agent's 1st training session, the final trained agent (as a pickle file or weights file), and three to five stages in-between.
Code quality, including clear structure, meaningful variable/function names, appropriate use of comments, and well-written docstrings.
Your submission should include a program that loads the fully trained agent and runs a short demo.

