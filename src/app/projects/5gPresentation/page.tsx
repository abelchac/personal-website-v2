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
      threshold: 0.9,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  let viewerElement = <InView as="div" onChange={(inView, entry) => {
    entry.target.children[0].classList.add('show'); 
     if (inView){
      console.log(entry.target.children[0].getElementsByClassName('video'));
      (entry.target.children[0].getElementsByClassName('video')[0]! as HTMLMediaElement).play();
    }
  }}
    
    
    className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}><div ref={ref}>{element}</div></InView>
  
  return viewerElement
}



export default function FiveGProject() {
  let title = (<>
  <h1>vRAN/ORAN/5G: Introduction and Downlink/Uplink Synchronization</h1>
  </>);


  let oran = FadeInFactory(<>
    <video className="video" controls src="5gPresentation/TradRanToOran.mp4"></video>
  </>);
  let oranZoom = FadeInFactory(<>
    <video className="video"  controls src="5gPresentation/AirInterfaceZoom.mp4"></video>
    </>);

  let frameStructure = FadeInFactory(<>
    <video className="video"  controls src="5gPresentation/Numerology_sync_bandwidth.mov"></video>
    </>);

  let tdd = FadeInFactory(<>
    <video className="video"  controls src="5gPresentation/Tdd_patterns.mp4"></video>
    </>);
  
  let ofdm = FadeInFactory(<>
    <video className="video"  controls src="5gPresentation/OFDM_Intuition.mp4"></video>
    </>);

  
  let resourceGrid = FadeInFactory(<>
    <video className="video"  controls src="5gPresentation/ResourceGrid.mp4"></video>
    </>);


    
    
  return (
    <>
    {title}
    {oran}
    {oranZoom}
    {frameStructure}
    {tdd}
    {ofdm}
    {resourceGrid}
    <section className="">
    
    </section>     
    </>
    );
}

