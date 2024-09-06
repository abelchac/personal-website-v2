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




export default function RAM_Project() {
  let design_problem = (<div>
    <h1>RAM Design</h1>
    <h2>Design Problem: Design a 16x4 (16 words of 4 bits) SRAM memory</h2>
    <section>
  
    <ul>
      <li>The design must have full word (4 bits) random access read and write capability and be SRAM, but the topology of the core cells and periphery circuits are open and available for optimization.</li>
  <li>Target technology is the High Performance 22nm process (/home1/e/ese3700/ptm/22nm HP.pm)</li>
  <li> Vdd ≤ 1V</li>
  <li> Design should be synchronous with the max CLK frequency (ie. min CLK period)
  based on your design’s delay (ie. write access time and read access time). Your max
  frequency must be greater than 500 MHz (ie. your memory must be able to operate
  with at least a 500 MHz clock).</li>

  <li> We will concretely focus on an SRAM memory array with capacity of 16, holding
  4b-wide data elements.</li>
  <li> Inputs: 4-bit address, active high write enable (WE=1 for write, WE=0 for read),
  4-bit data bus, and a single clk signal for synchronization</li>
  <li> Outputs: 4-bit data bus</li>
  <li> A read or write operation must be completed within one clock period</li>
    </ul>
    </section>

  </div>)

  return (
      <section className="">
      {design_problem}
      </section>     
     
    );
}

