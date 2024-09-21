"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { CodeBlock, dracula } from "react-code-blocks";
import { useInView, InView} from "react-intersection-observer";


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


export default function ShotChartProject() {



let video_section = FadeInFactory(
  <video  controls src="nba_shot_chart/shot_chart_demo.mp4"/>);
let section1 =  (<div className=""><h1>NBA Shot Chart Visualizer</h1>
  <Link href="https://github.com/abelchac/Nba-Shot-Charts"> <h2>Project Link: https://github.com/abelchac/Nba-Shot-Charts</h2></Link>
  </div>);



let section2 = FadeInFactory(  <div>
  <br></br>
  <p>NBA_SHOT_CHART produces a visualization of player or team 
shot charts. Selection of player or team can be done 
through the UI. As well as being able to choose the range
of shots of either 8ft or 5ft. There are a total of 3 plots for
visualization: Shot Chart on basketball court plot, 
Shot Attempts of Selected vs League Average Attempts, and 
Average of Selected vs Leauge Average. There are tkinter option
menus for selecting the distance ranges of the shots and whether a player or 
team will selected. The plot which contains the basketball court visualization
plots wedges with either green or red, with the opacity of the color representing
how far from the average the player is (greater the opacity the futher from average). </p>
<br></br>
<p>
The data for the shot charts is produced by using the
nba_api library (https://github.com/swar/nba_api) and will
be parsed using pandas. The data will be held peristently through
pickle as not to time out the api. The visualization and UI will be 
made by using tkinter, matplotlib, and seaborn. </p>
<img src="nba_shot_chart/example.png"></img>
</div>
);

let section3_text = `tkinter  
pickle  `

let section3 = FadeInFactory(<div><h2>First-party modules:</h2>
<br></br>
 <CodeBlock text={section3_text} language="python" theme={dracula} showLineNumbers={false}></CodeBlock>
 </div>)

let section4_text = `pandas  
matplot_lib  
nba_api  
`
let section4 = FadeInFactory(<div><h2>Third-part modules:</h2>
<br></br>
 <CodeBlock text={section4_text} language="python" theme={dracula} showLineNumbers={false}></CodeBlock>
 </div>)

let section5_text = `pip install nba_api  
pip install pandas  
pip install matplotlib`
let section5 = FadeInFactory(<div>
  <h2>Installation</h2>
  <h3>Requirements:</h3>
  <CodeBlock text={section5_text} language="python" theme={dracula} showLineNumbers={false}></CodeBlock>
</div>)


let section6_2_text = `Handles Displaying UI elements and tkinter windows. The values 
for the selected player/team and shot range is adjusted within this file.
The players and teams are listed within a listbox with the data visualized 
with matplotlib and seaborn plots to the left and right.  `;
let section6_3_text = `Credit to http://savvastjortjoglou.com/nba-shot-sharts.html for code for 
visualizing a basketball court in matplot lib that is functional with 
data provided by nba_api.  `;
let section6_4_text = `Creates the matplotlib elements (wedges) for displaying on the basketball court axes.  `;
let section6_5_text = `Class for generating averages and shot attempts that remain constant throughout the
life time of the program. The magic methods are the __init__ for the initialization
of the class and __getitem__ to make the class subscriptable making the getting of the
averages within the drawing phase easier as there is no need to shuffle through
different arrays as the data will be preset with the class object.  `;
let section6_6_text = `Within the file the functions are for getting information of the nba players and teams.  `;
let section6_7_text = `Contains function for getting the shot data of a single player or team.  `;



let section6_1 = FadeInFactory(<h2>Code Strucutre</h2>);
let section6_2 = FadeInFactory(<div><h3>UiMain.py</h3>
<CodeBlock text={section6_2_text}  theme={dracula} showLineNumbers={false}></CodeBlock>
</div>);
let section6_3 = FadeInFactory(<div><h3>draw_court.py</h3>
<CodeBlock text={section6_3_text}  theme={dracula} showLineNumbers={false}></CodeBlock>
</div>);
let section6_4 = FadeInFactory(<div><h3>draw_ranges.py</h3>
<CodeBlock text={section6_4_text}  theme={dracula} showLineNumbers={false}></CodeBlock>
</div>);
let section6_5 = FadeInFactory(<div><h3>playerTeamClass.py</h3>
<CodeBlock text={section6_5_text}  theme={dracula} showLineNumbers={false}></CodeBlock>
</div>);
let section6_6 = FadeInFactory(<div><h3>team_or_players.py</h3>
<CodeBlock text={section6_6_text}  theme={dracula} showLineNumbers={false}></CodeBlock>
</div>);
let section6_7 = FadeInFactory(<div><h3>get_shots.py</h3>
<CodeBlock text={section6_7_text}  theme={dracula} showLineNumbers={false}></CodeBlock>

</div>);


let section7_text = `python ./UiMain.py
Select the desired ZoneDistance from the ZoneDistance option menu,
do the same for selecting if you would like a team or a player. 
After those two selection are made, pick a player or team that 
you would like to have their data visualized. `
let section7 = FadeInFactory(<div><h2>Usage</h2>

<br></br>
<CodeBlock text={section7_text} theme={dracula} showLineNumbers={false}></CodeBlock>

</div>);


return ( <section> 

  {section1}
  {video_section}
  {section2}
  {section3}
  {section4}
  {section5}


<div>
{section6_1}
<br></br>
{section6_2}
<br></br>
{section6_3}
<br></br>
{section6_4}
<br></br>
{section6_5}
<br></br>
{section6_6}
<br></br>
{section6_7}
<br></br>
{section7}
</div>
<br></br>



</section>);
}

