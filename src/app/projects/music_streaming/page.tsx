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



export default function MusicStreamingProject() {
  let project_description = <div>
    <h1>Streaming Music Service</h1>
    <p> In this project, we deisgned and implemtned a protocol of our own for a music streaming service.
      We had to have a server with music files respodning to a client request for music.
    </p>
    <h2>Requirements</h2>
    <section>
      <ul>
        <li>You will turn in an RFC for your protocol that specifies the exact behavior of your protocol.</li>
        <li>Your implementation should be directly on top of raw sockets. You should not, for instance, use any existing HTTP or RPC implementation.</li>
        <li>Your server should be able to handle multiple clients simultaneously. They should be able join and leave at any time, and the server should continue to operate seamlessly.</li>
        <li>Your client should be interactive and it should know how to handle at least the following commands:</li>
        <section>
        <ul>
          <li>list: Retrieve a list of songs that are available on the server, along with their ID numbers.</li>
          <li>play [song number]: Begin playing the song with the specified ID number. If another song is already playing, the client should switch immediately to the new one.</li>
          <li>stop: Stops playing the current song, if there is one playing.</li>
        
        </ul>

        </section>
        <li>Similar to the above, the server should be able to handle new commands without needing to finish sending the previous file. For instance, if it receives a list command during playback of a giant music file, the client should be able to see the list of songs immediately.</li>
        <li>The client should not cache data. In other words, if the user tells the client to get a song list or play a song, the two should exchange messages to facilitate this. Don't retrieve an item from the server once and then repeat it back again on subsequent requests.</li>
      </ul>

    </section>
  </div>

  return (
      <div>
        {project_description}
        <section className="">
       <object className="pdf" 
            data="music_streaming/rfc.html">
    </object>
      </section>     
      </div>
      
     
    );
}

