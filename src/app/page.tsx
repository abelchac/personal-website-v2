"use client";
import Image from "next/image";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";


function ProjectCard(props : {project_svg_name : string, alt_text : string}){
  let data_string = './cards/' + props.project_svg_name;
  
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.3,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  if (inView) {
    entry?.target.classList.add('show');
  } else{
    entry?.target.classList.remove('show');
  }
  console.log(inView);
  

  return (
    <InView as="div" onChange={(inView, entry) => {entry.target.classList.add('show')}}>
      <img ref={ref} className={`place-items-center ProjectCard transition-opacity duration-1000 ${inView ? 'opacity-100' : 'opacity-0'}`} src={data_string} alt={props.alt_text}/>
    </InView>
  );
}


export default function Home() {

  //let card_csgo = create_card("CSGO_CARD.svg");

  return (
    
  <span>
  <div>
    <div className="grid grid-cols-[repeat(auto-fill,minmax(10rem,25rem))] gap-x-16 gap-y-10 border-2 p-8 justify-center">
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
      <ProjectCard project_svg_name="CSGO_CARD.png" alt_text=""/>
    </div>
  </div>
    <ProjectCard project_svg_name="CSGO_CARD.png" alt_text="Cards that looks like a trading card that has the title: 'CSGO WIN Predictor', 
    attributes: Big Data, Machine Learning, Data Visualization. 
    Description: Using feed forward neural networks and CSGO player statistics, a game result was predicted "/>

    
 
  </span> 
    
  );
}

