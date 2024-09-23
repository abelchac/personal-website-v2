"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";
import { ProjectCard } from "@/components/Card";

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




export default function Home() {

  //let card_csgo = create_card("CSGO_CARD.svg");

 
  let csgo_card_alt_text = `Cards that looks like a trading card that has the title: 'CSGO WIN Predictor', 
    attributes: Big Data, Machine Learning, Data Visualization. 
    Description: Using feed forward neural networks and CSGO player statistics, a game result was predicted `;

  let bicycle_gan_alt_text = `
  attributes: Machine Learning, Big Data, Research
  Description:
  Implemented BicycleGAN which is an extension of CycleGAN that links basic GAN and VAE models in order to map an edge image into multiple RGB images
  `
  let kd_alt_text = `
  attributes: Data Structures and Algorithims, Data Analytics, Image Processing :
  Description: Using K-d trees, compressed the color space of an image while minimizing the variance between colors`
  let nba_awards_alt_text = `
  attributes: Data Structures and Algorithims, Web Scraping, Object Oriented
  Description: Player awards are predicted using polynomial functions based on the current NBA stats`
  let lyric_gen_alt_text = `
  attributes:  Big Data, Machine Learning, API Usage
  Description: Applied various machine learing models to the task of generating lyrics given a starting phrase`;
  let nba_shot_chart_alt_text = `
    attributes: Object Oriented, API Usage, Data Visualization
  Description: Produces a visualization of player or team shot charts and compares the charts to league avgerage`;
  let ram_design_alt_text = `
    attributes: Circuit Design, Circuit Simulation, Circuit Optimization
  Description: Using Electric and NgSpice, designed a 16 x 4 bit RAM array and tested read and write up 1GHZ`;
  let music_streaming_alt_text = `
    attributes: Networking, Object Oriented, Multithreading
  Description: Music Streaming Protocol, MSP, is an application-level protocol for real-time control over the delivery of audio data, e.g. songs.`;
  let snake_ai_alt_text = `
    attributes: Data Structures and Algorithims,  Machine Learning, Q-Learning
  Description: A naive approach to solving the game of snake using a Deep Q-Learning Model.`;
  let adder_alt_text = `
    attributes: Circuit Design, Circuit Simulation, Circuit Optimization
  Description: Using Electric and NgSpice, designed and tested an 8-bit adder with several varation to the design  `

  let cards = [ <ProjectCard key={0} project_svg_name="KdTreesCard.png" alt_text={kd_alt_text} href="kd_trees"/>,
    <ProjectCard key={1} project_svg_name="NbaAwardsCard.png" alt_text={nba_awards_alt_text} href="nba_awards_predictor"/>,
    <ProjectCard key={2}  project_svg_name="BicycleGan.png" alt_text={bicycle_gan_alt_text} href="bicyclegan"/>,
    <ProjectCard key={3} project_svg_name="LyricGenAI.png" alt_text={lyric_gen_alt_text} href="lyric_gen"/>,
    <ProjectCard key={4} project_svg_name="NbaShotChartCard.png" alt_text={nba_shot_chart_alt_text} href="nba_shot_chart"/>,
    <ProjectCard key={5} project_svg_name="RamCard.png" alt_text={ram_design_alt_text} href="ram_design"/> ,
    <ProjectCard key={6} project_svg_name="MusicStreamingCard.png" alt_text={music_streaming_alt_text} href="music_streaming"/>,
    <ProjectCard key={7} project_svg_name="CSGOCARD.png" alt_text={csgo_card_alt_text} href="csgo_predictor"/>,
    <ProjectCard key={8} project_svg_name="SnakeAICard.png" alt_text={snake_ai_alt_text} href="simple_dqn" />,
    <ProjectCard key={9} project_svg_name="AdderCard.png" alt_text={adder_alt_text} href="adder" />]

  
    const { ref, inView, entry } = useInView(
      {
        /* Optional config */
        threshold: 0.1,  // Trigger when 10% of the element is in view
        triggerOnce: true,  // Trigger only once
      }
    );
  
    if (inView) {
      entry?.target.classList.add('shown');
    } else{
      entry?.target.classList.remove('shown');
    }
    
    let project_title = FadeInFactory(<div><h1 className="center ProjectTitle">Projects</h1></div>);
  return (
    
  <section>
     
     <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
      <div ref={ref}>
        <p>
          Hi, I'm Abel Chacko. I'm currently working as a Platform Application Engineer at Intel. My role entails enabling customers to use our FlexRAN software for their RAN solutions. 
          Helping them solve issues from system enablement to setting up a full end-to-end 5G stack.
          I graduated from the University of Pennsylvania in 2022 with my Master's in Robotics and Bachelor's in Computer Engineering. 
          
          I'll finish this description later
        </p>
          
      </div>    
      <br/>
      <br/>
      <br/>
      <br/>
      <br/>
      <br/>
      </InView>
    <section>
    <br></br>
    {project_title}
    <div className="projectCardGrid grid grid-cols-[repeat(auto-fill,minmax(6rem,25rem))] gap-x-16 gap-y-10 justify-center" >
    {cards[0]}
    {cards[1]}
    {cards[2]}
    {cards[3]}
    {cards[4]}
    {cards[5]}
    {cards[6]}
    {cards[7]}
    {cards[8]}
    {cards[9]}

    </div>  
    </section>   
  </section> 
    
  );
}

