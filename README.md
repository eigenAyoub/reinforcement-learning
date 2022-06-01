I'll collect in this Repo my personal **programming** work that I have done for some RL courses that I have followed. I was mainly following, Amir Massoud  Fahramand's course []() that was taught in the University of Toronto. To me, it was a perfect balance between theory and application. Contained proofs of almost every result (e.g., proof of convergence of the main algorithms, proofs that "some operator" is a contraction mapping besides the known ones like the Bellman Op, etc.)

I'll quote the official description of the course:

>This is an introductory course on reinforcement learning (RL) and sequential decision-making under uncertainty with an emphasis on understanding the **theoretical foundation**. We study how dynamic programming methods such as value and policy iteration can be used to solve sequential decision-making problems with known models, and how those approaches can be extended in order to solve reinforcement learning problems, where the model is unknown. Other topics include, but not limited to, function approximation in RL, policy gradient methods, model-based RL, and balancing the exploration-exploitation trade-off. The course will be delivered as a mix of lectures and reading of classical and recent papers assigned to students. As the emphasis is on understanding the foundation, you should expect to go through mathematical detail and proofs. Required background for this course includes being comfortable with probability theory and statistics, calculus, linear algebra, optimization, and (supervised) machine learning.


As I mentioned above, this Repo will mainly include programming related aspects. In my [webpage](https://eigenayoub.github.io/), I might write on some theoritical RL aspects that were part of the course/assignements.

### ToC:
* [Repo structure](#repo-structure)
* [Credits](#credits)
* [References](#references)


### Repo structure

* Part 1: Dynamic Programming   [DONE]
  * Value Iteration
  * Policy iteration
  * Q-learning

* Part 2: Value Approximation / Deep RL  [DONE]
  * Overestimating bias in Q-learning   [DONE]
  * Convergence of TD   [DONE]
  * Implementing DQN   [DONE]
  * Reporting [TO-DO]
  * Implementing the DDQN loss  [TO-DO]

* Part 3:  Gradient-Based Methods 
  * Polcicy Gradient [DONE]
    * Gaussian Policy Network [Done]
  * Actor Critic [TO-DO]


### Credits (so far):
* [The main course](https://amfarahmand.github.io/IntroRL/)
* [DeepMind x UCL, RL course, version 2021, Hado van Hasselt](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)
* [Standord, CS234: Reinforcement Learning Winter 2022](https://web.stanford.edu/class/cs234/)

### References:
* [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book.html)
* [Csaba Szepesvári](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
* [Lecture notes of the main course](https://amfarahmand.github.io/IntroRL/lectures/LNRL.pdf)