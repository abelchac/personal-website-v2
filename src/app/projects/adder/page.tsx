"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";
import { CodeBlock, dracula } from 'react-code-blocks';



function fadeInFactory(element : React.JSX.Element){
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



export default function AdderProject() {

  let design_problem = (<div>
    <h1>RAM Design</h1>
    <h2>Design Problem: Design the circuit-level implementation of a full-adder bit-slice (for a
      ripple-carry adder).</h2>
    <section>
  
    <ul>
      <li>Your design must be cascadable to build adders of arbitrary bit width and usable in
      adder trees (e.g. as used in multipliers).</li>
  <li>Target technology is the High Performance 22nm process</li>
  <li> Vdd ≤ 1V</li>
      </ul>
    </section>

  </div>)

  return (
    <div>
    {design_problem}
       <object className="pdf" 
            data="adder/adder.pdf">
    </object>
    </div>
    );
}

