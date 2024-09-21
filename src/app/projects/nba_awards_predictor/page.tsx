"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";
import { CodeBlock, dracula } from 'react-code-blocks';
import { start } from "repl";



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



export default function NbaAwardsProject() {
  let title = (<div className=""><h1>Nba Awards Predictor</h1>
    <Link href="https://github.com/abelchac/nba_awards_predictor"> <h2>Project Link: https://github.com/abelchac/nba_awards_predictor</h2></Link>
    </div>);

  let description = FadeInFactory(<div>
    <h2>Description:</h2>
    <p>Description: We are interested in the NBA and particularly how different awards are given out. As a result,
designed a program that takes into the account the statistics of each player in the league and calculates
a score using the data. We begin with a table that displays each player and their statistics. From there, 
depending on the award chosen (MVP, ROTY, DPOY), uses different statistics to narrow down candidates for
each award and show which players have the best statistics for the award. Furthermore, the weight that 
each statistic carries in the calculations can be modified to best fit the user&apos;s preference for which 
statistics that matter most. We also have a MVP-BFS function that finds a path of players, beginning with
an MVP and through their teammates, creating a path to another MVP. All in all, we have designed a
program that compiles the statistics of all current NBA players in a table. From there, we can narrow down
the table based on whichever award is chosen by completing calculation with their statistics to obtain an
award score, which is automatically sorted to list the best players fit for that award. Finally, we have 
also included an additional function that finds the shortest path between one MVP to another MVP using 
teammates. </p>
  </div>);

  let relevant_cat = FadeInFactory(<div>
    <h2>Relevant categories:</h2>
    <p>Our projects uses Document Search or Information Retrieval techniques to find and 
store each player&apos;s statistics in a table. This is seen in several places. To compile the main table, we
retrieve information from <Link href="https://www.nba.com/stats/"> https://www.nba.com/stats/</Link> and store them as JSON Objects. The second place
we use this is in the MVP-BFS function, where we obtain teammates for each MVP and their teammates as
well, using the base link: <Link href="https://basketball.realgm.com/">https://basketball.realgm.com/</Link>, and modifying it to obtain a new link and 
getting additional information there.</p>
<br></br>
<p>Also in the MVP-BFS section, we use Graph and Graph Algorithm concepts. We first a build a graph from the
MVPs, their teammates, and teammates of those teammates. We then use BFS to find a path from one MVP and 
following their teammates and teammates of those teammates to another MVP.
</p>
  </div>);
    
  let startScreen = FadeInFactory(<div>

    <h2>Start the Program:</h2>
    <h2>Initial Screen:</h2>
    <p>The base table shows all current players in the NBA with all traditional statistics</p>
    <Image className="!relative" fill  src="/nba_awards_predictor/start.png" alt={""}></Image>
  </div>);

  let buttons = FadeInFactory(<div>

    <h2>Buttons:</h2>
    <Image className="!relative" fill  src="/nba_awards_predictor/buttons.png" alt={""}></Image>
    <p>The buttons to the right of the table allow the user to select which actions to take.</p>
    <p> MVP (Most Valuble Player): Will open up the MVP predictor screen</p>
    <p>DPOY (Defensive Player Of The Year): Will open up the DPOY predictor screen</p>
    <p>ROTY (Rookie Of The Year): Will open up the ROTY predictor screen</p>
    <p>MVP-BFS: Will open up the MVP-BFS screen</p>
    <p>RESET: Reset the tables and take the user back to the initial screen</p>
  </div>);    
  
  let mvp = FadeInFactory(<div>

    <h2>MVP </h2>
    <p>The MVP predictor screen will appear like this with initial weights on the sliders set. These sliders can be
changed and the weight changes will be reflected on the sliders and the MVP score function. These values
will compute new MVP Scores which are placed into the MVP score column for each player. This column
is auto sorted so that the player with the highest score is at the top.</p>
<Image className="!relative" fill  src="/nba_awards_predictor/mvp.png" alt={""}></Image>
  </div>);
  
let mvpSliders = FadeInFactory(<div>

  <h2>MVP SLIDERS</h2>
  <p>To select the weights for the MVP score function, use the sliders that are to the right of
the table if you wish to adjust how player statistics should be reflected in an MVP
candidate.</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/mvp_sliders.png" alt={""}></Image>

</div>);
  
let mvp_score = FadeInFactory(<div>

  <h2>MVP SCORE FUNCTION
  </h2>
  <p>The MVP SCORE function takes the weights from the sliders and changes the label of
  present in the frame, so that the user can see how all the weights apply.</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/mvp_score.png" alt={""}></Image>

</div>);

let dpoy = FadeInFactory(<div>

  <h2>DPOY</h2>
  <p>The DPOY predictor screen will appear like this with initial weights on the sliders set. These sliders can
be changed and the weight changes will be reflected on the sliders and the DPOY score function. These
values will compute new DPOY Scores which are placed into the DPOY score column for each player.
This column is auto sorted so that the player with the highest score is at the top.</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/dpoy.png" alt={""}></Image>

</div>);

let dpoy_sliders = FadeInFactory(<div>

  <h2>DPOY SLIDERS
  </h2>
  <p>To select the weights for the DPOY score function, use the sliders that are to the right of
the table if you wish to adjust how player statistics should be reflected in an DPOY
candidate</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/dpoy_sliders.png" alt={""}></Image>

</div>);

let dpoy_score = FadeInFactory(<div>

  <h2>DPOY SCORE FUNCTION</h2>
  <p>The DPOY SCORE function takes the weights from the sliders and changes the label of
  present in the frame, so that the user can see how all the weights apply.</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/dpoy_score.png" alt={""}></Image>

</div>);

let roty = FadeInFactory(<div>

  <h2>ROTY</h2>
  <p>The ROTY predictor screen will appear like this with initial weights on the sliders set. These sliders can
be changed and the weight changes will be reflected on the sliders and the ROTY score function. These
values will compute new ROTY Scores which are placed into the ROTY score column for each player.
This column is auto sorted so that the player with the highest score is at the top.
</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/roy.png" alt={""}></Image>

</div>);

let roty_sliders = FadeInFactory(<div>

  <h2>ROTY SLIDERS
  </h2>
  <p>To select the weights for the ROTY score function, use the sliders that are to the right of
the table if you wish to adjust how player statistics should be reflected in an ROTY
candidate.</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/roy_sliders.png" alt={""}></Image>

</div>);

let roty_score = FadeInFactory(<div>

  <h2>ROTY SCORE FUNCTION</h2>
  <p>The ROTY SCORE function takes the weights from the sliders and changes the label of
  present in the frame, so that the user can see how all the weights apply</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/roy_score.png" alt={""}></Image>

</div>);

let mvp_bfs = FadeInFactory(<div>

  <h2>MVP-BFS</h2>
  <p>Select a former MVP winner from each of the 2 lists and press find path to generate the shortest
path of teammates that will go from one MVP to the other. Without the playerGraph.ser file, the
MVP-BFS will not be able to run until the graph is fully scrapped from
<Link href="https://basketball.realgm.com/">https://basketball.realgm.com/</Link>. Therefore, we have included the playerGraph.ser file so that the
user will not have to wait for the graph to be built to be able to use the MVP-BFS functionality.
However, all other aspects of the program will be full functionally because of the MVP-BFS
graph being built on another thread</p>
  <Image className="!relative" fill  src="/nba_awards_predictor/mvp_bfs.png" alt={""}></Image>

</div>);


  return (
      <section className="">
        {title}
        <section>
        {description}
        <br></br>
        {relevant_cat}
        </section>
        <br></br>
        {startScreen}
        <br></br>
        {mvp}
        <section>
          {mvpSliders}
          {mvp_score}
        </section>
        <br></br>
        {dpoy}
        <section>
          {dpoy_sliders}
          {dpoy_score}
        </section>
        <br></br>
        {roty}
        <section>
          {roty_sliders}
          {roty_score}
        </section>
        <br></br>
        {mvp_bfs}
      </section>     
     
    );
}

