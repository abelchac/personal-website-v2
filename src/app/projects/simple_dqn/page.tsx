"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";
import { CodeBlock, dracula } from 'react-code-blocks';



function FadeInFactory(element : React.JSX.Element){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  let viewerElement = <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}><div ref={ref}>{element}</div></InView>
  
  return viewerElement
}



export default function SimpleDQN() {

  let introduction = FadeInFactory(<div>
    <h2>1. INTRODUCTION</h2>
    <p>The composition of video games lends itself to being a common domain for research in machine learning. The
environments with consistent constraints and rules allows for researchers to develop and tune agents to learn based on
the gameplay in a reasonable fashion. Often for game developers the goal in these games was to develop agents which
simulate intelligence, learning from past experience within the game. The idea of finding the optimal strategy purely
with a dynamic programming approach may work in a small domain, but once the structure becomes more complex
other unfixed strategies must be applied [1]. </p>
<br></br>

<p>Though reinforcement learning has been applied to other games and supposed human-level performance. This was
applied to the game of backgammon and led to the model of TD-gammon. With the breakthrough of DeepMind to
combine reinforcement learning and deep learning coming up with the model of deep Q-Learning network (DQN)
models, the foundation of a generalizable model began to take root [2].</p>
<br></br>

<p>The implementation of a DQN model however is not completely naive. The issue arises when developing a reward system for the agent [2]. Rewards may not always be present and the effect of the reward which can be delayed
can lead to the learning of policies that do not play a game in the correct manner. The game of snake is about a single
player that moves around a grid attempting to eat as many pieces of food as possible, without running into the bounds
or themselves. The difficulty of the game progressively gets harder, once the player eats a piece of food, the snake will
get longer forcing the player to route around the body of the snake.</p>
<br></br>

<p>Thus it is important to distribute many samples within the snake agent’s learning. The learning must come from
a variety of states with the snake being both short and long, and with food items in many different positions. To do
this a replay buffer is necessary to remember the previous episodes and iterations of the snake. The use of the replay
buffer allows for the generation and training on uncorrelated data, enabling the agent to learn about many game play
experiences [3].</p>
  </div>)


  let background = FadeInFactory(<div>
    <h2>2.1. Environment.</h2>
    <p>The environment of the snake game lies within a 10x10 grid with each cell either being filled with a 0, 1, or 2. Each of
the numbers represent a different state of the cells, 0 being that the cell is empty, 1 the snake is occupying the cell, 2 the
food item is present at the cell. For simplicity the snake agent has the same starting position for all new environments,
but the initial position of the food item is a random location that is empty. </p>
<br></br>
<p>The snake is able to move in 4 different directions: UP, DOWN, LEFT, and RIGHT. The snake also has heading, which is initially set to RIGHT. The snake cannot go into the opposite direction of the heading as this would cause
the snake to run into itself. The episode for the snake game will either end in 1000 iterations or until the snake runs into
an illegal cell. If the agent runs into a food cell the size of the snake will increase by one, and another piece of food will
be placed into a random empty cell</p>
    <figure>
      <img   src="simple_dqn/snake.png" alt={""}></img >
      <figcaption>FIGURE 1. A possible initial state of the environment. Green is the snake, yellow is the food, and
      purple are the empty cells</figcaption>
    </figure>
    </div>)


  let related_work = FadeInFactory(<div>
    <h2>3. RELATED WORK</h2>
    <p>There exist several works that pertain to the exact topic of solving the game of snake within a grid space using
different forms of reinforcement learning models. All the works reviewed use an ε-greedy strategy as a way creating an
exploratory model which will not always follow the path with the greatest probability, but rather with probability of ε,
will take a random action.</p>
<br></br>
<p>One of the works instead of starting from a grid state, take in image inputs into the respective networks for the
state. Learning from image states rather than knowledge that is known as ground fact.
</p>
<br></br>
<p>Furthermore, two methods are laid out for the process in which the snake agent learns and is rewarded. At each
iteration the DQN stores experiences that a created with the output action of the network storing the state, action,
reward and next state in a tuple. One of the approaches has the Q-value for the current state and action be a cumulative value based a recursive formula on the previous states and actions with a discount factor at each recursive layer [2].</p>
<br></br>
<p>The other approach was to have several components building up to the reward. The components were the distance
reward, a training gap, and timeout strategy. The distance reward increases the reward if the action causes a moves towards the
target and decrease if it shifts away; the training gap will temporarily remove experiences from learning if they are not
suitable, such as right after an agent is done eating food; the timeout strategy will give a negative reward if the agent
goes over a specified number of steps without eating a food cell [1].</p>

  </div>)

  let approach_code = `Initialize snake board
  While NOT(DONE)
  u := Q(s) with epsilon weight
  apply action u to board
  Q(s\') := r\' + Q(s)
  `

  let approach  = FadeInFactory(<div>

    <p>The snake agent is initialized with a randomized weights according to a normal distribution. Then before any training
begins with an ε of 1, 1000 episodes of the snake environment are run until completion. These are the initial episodes
that are to be use in the replay buffer.
</p>
<p>Each game of Snake runs in the following algorithmic way</p>
<CodeBlock text={approach_code} language="python" theme={dracula}></CodeBlock>
<p>The there is a cumulative sum of the previous rewards which is carried to the Q-value of the next state</p>

<section>
<table className="dataframe w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400 text-lg">
  <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
    <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
      <th>Case</th>
      <th>Food</th>
      <th>Hit Self</th>
      <th>Hit Wall</th>
      <th>Move</th>
    </tr>
  </thead>
  <tbody>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>Reward</th>
      <td>+500</td>
      <td>-100</td>
      <td>-100</td>
      <td>-10</td>
    </tr>
    </tbody>
  </table>
  </section>

  <p>The DQN will only ever enter a learning step if a total of 1000 game steps have passed. Giving the network time to
build up experiences before learning again. Then the loss is calculated basted on the Q-value of the experience versus
the target Q-value computed from a delayed target network. The epsilon at each iteration, which began at 1, is decayed
by a factor of .999 with a minimum value of .002 for epsilon, so that we drastically decrease the number of random
actions as learning progresses. Also, if the average number of moves for each episode after 1000 game steps is 30%
less than the maximum, a weighted average of the network that produced the largest average number of moves and the
current network is taken. The weighting is 9 parts the current network and 1 part the maximum average steps network.</p>

  </div>)

  let experimental_results = FadeInFactory(<div>
    <h2>5. EXPERIMENTAL RESULTS</h2>
    <p>Unfortunately this model was unable to sufficiently learn to survive or target food cells to increase in size. There
are many reasons and possible points of improvement to be discussed in the following section. Figures 2 and 3, show
training data for how many iterations of new snake games had to be conducted to reach 100,000 game steps. The
training occurring every 1000 game steps, did not provide much benefit to teach the agent a policy that was beneficial to
increasing the size of the snake nor to surviving a long duration of time. The average does increase in some areas, but
the number of moves would settle down to a much lower number. Then within the evaluation which was conducted over
1000 games of snake, produced the same number of average moves for all conditions. This is probably due to the initial
snake position being the same for all states and epsilon being 0 allowing the snake with the incorrect policy to move to
its death in 3 steps. The average food eaten is negligible and most likely only because of the random placement of food.</p>
  <div className="grid grid-cols-[repeat(auto-fill,minmax(6rem,25rem))] gap-x-16 gap-y-10 justify-center">

  <figure>
  <img   src="simple_dqn/results.png" alt={""}></img >
  <figcaption>FIGURE 2. A rolling average of number of moves in each iterations versus the total number of
iterations in training
</figcaption>
  </figure>
  
  <figure>
  <img   src="simple_dqn/foodVsIteration.png" alt={""}></img >
  <figcaption>FIGURE 3. A rolling average of number of food eaten in each iterations versus the total number of
  iterations in training
</figcaption>
  </figure>

  <figure>
  <img   src="simple_dqn/eval.png" alt={""}></img >
  <figcaption>FIGURE 4. A rolling average of number of moves in each iterations versus the total number of
  iterations in evaluation
</figcaption>
  </figure>

  <figure>
  <img   src="simple_dqn/eval2.png" alt={""}></img >
  <figcaption>FIGURE 5. A rolling average of number of food eaten in each iterations versus the total number of
iterations in evaluation
</figcaption>
  </figure>

  </div>
 

  </div>)

  let discussion = FadeInFactory(
    <div>
    <h2>6. DISCUSSION</h2>
    <p>The Simple DQN implementation with a simple replay buffer, played the game of snake at a worse rate than
a randomly chosen actions. There are many short comings to the design that was implemented above including,
unselective replay buffer, poor reward choice, poor reward computation, input data. These short comings and oversights
mentioned are probably not the only missing components, but to the author seem the most prevalent. The random
and indiscriminate selection of experiences of the replay buffer did not allow for the model to learn from states in
which food was eat, but rather because the amount of states where the snake immediately ran into a wall were more
abundant the model gained experience from these more often than not. Also the reward composition was implemented
poorly compared to the reviewed papers, which did not use hard values always for the reward but rather would scale
the rewards with either a scaling factor for each consecutive state or other components of the rewards which would
evolve from state to state. Finally the input data may have possibly played a factor, by feeding in a 10x10 grid and
the snake’s head post ion, could have given the network too much data that was similar to other states that it would
take a much longer time to train to differentiate between these states to being playing the game properly. Overall, this
implementation was a failure in many sense, but also enlightens one on the difficulty of developing a proper state space
and reward that are necessary in computing and developing a reinforcement-learning based algorithim.</p></div>

  )

  let reference = FadeInFactory(<div>
    <h1>REFERENCES</h1>
    <p>[1] A. J. Almalki and P. Wocjan, ”Exploration of Reinforcement Learning to Play Snake Game,” 2019 International Conference on Computational Science and Computational Intelligence (CSCI), 2019, pp. 377-381, doi:
    10.1109/CSCI49370.2019.00073.</p>
    <p>[2] Wei, Z., Wang, D., Zhang, M., Tan, A. H., Miao, C., Zhou, Y. (2018, July). Autonomous agents in Snake game via
    deep reinforcement learning. In 2018 IEEE International Conference on Agents (ICA) (pp. 20-25). IEEE.</p>
    <p>[3] Zhang, S., Sutton, R. S. (2017). A deeper look at experience replay. arXiv preprint arXiv:1712.01275.</p>


  </div>)
  

  return (
      <section className="">
      <h1>SIMPLE DQN: UNSOLID SNAKE</h1>
      <p>ABSTRACT. A naive approach to solving the game of snake using a Deep Q-Learning Model. This approach without properly
      biasing previous states or selecting more rewarding experiences leads to minimal gains to the overall size or lifetime of a snake
      agent. Even if a large negative reward is given for improper actions, the agent is unable to learn due to the sheer number of
      negative experience that it learns from.
      </p>
      <br></br>
      {introduction}
      <br></br>
      {background}
      <br></br>
      {related_work}
      <br></br>
      {approach}
      <br></br>
      {experimental_results}
      <br></br>
      {discussion}
      <br></br>
      {reference}
      </section>     
     
    );
}

