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



export default function KdTreesProject() {
  let title = (<div className=""><h1>KD-Trees Image Compression</h1>
    <Link href="https://github.com/abelchac/kdTreesColorCompression"> <h2>Project Link: https://github.com/abelchac/kdTreesColorCompression</h2></Link>
    <p>While working on another project, I got stuck trying to find a faster way to compress the color space of my images. That's when I ran into this blog post <Link href="https://www.crisluengo.net/archives/932/">https://www.crisluengo.net/archives/932/</Link>
    by Chris Luengo. The concept is to use KD-Trees and <Link href="https://en.wikipedia.org/wiki/Otsu%27s_method">Otsu's Algorithm Iteratively</Link> to maximize the variance that we can achieve using a few colors as possible.</p>
    </div>);


  return (
      <section className="">
      {title}
      </section>     
     
    );
}

