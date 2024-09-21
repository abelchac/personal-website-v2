
import { useInView, InView} from "react-intersection-observer";

import "../app/globals.css";
import React from "react";
import Image from "next/image";


export function ProjectCard(props : {project_svg_name : string, alt_text : string, href : string}){
  let data_string = '/cards/' + props.project_svg_name;
  
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.05,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  if (inView) {
    entry?.target.classList.add('show');
  } else{
    entry?.target.classList.remove('show');
  }
  //console.log(inView);
  let href = "~/projects/" + props.href;
  return (
    <InView as="div" onChange={(_, entry) => {entry.target.children[0].classList.add('shown')}} className={`transition-opacity duration-1000 zoom hid ${inView ? 'opacity-100 shown' : 'opacity-0'}`} onMouseMove={mouseMove} onMouseOut={mouseOutFun} >
      <a href={href} >
      <Image loading="eager" fill ref={ref} className={`place-items-center ProjectCard !relative`} src={data_string} alt={props.alt_text}  />
      </a>
    </InView>
  );
}

function clamp(number: number, min: number, max: number) {
  return Math.max(min, Math.min(number, max));
}

function mouseOutFun(e : any) {
  let card = e.target

  card.style.transform = `rotateX(0deg) rotateY(0deg)`
}

const mouseMove = (e: any) => {
  //console.log('touch')
  //console.log(e.clientX);
  //console.log(e.clientY);
  
  let card = e.target;


  let bound = card.getBoundingClientRect();
  const cardWidth = card.offsetWidth * 1.5; 
  const cardHeight = card.offsetHeight * 1.5;
  const cardWidthHalf = cardWidth/2;
  const cardHeightHalf = cardHeight/2;
  const centerX = bound.left + cardWidthHalf;
  const centerY = bound.top + cardHeightHalf;


  const mouseX = e.clientX - centerX ;
  const mouseY = e.clientY - centerY;

  const rotateX = clamp(25*mouseY/(cardHeightHalf), -30, 30);
  const rotateY = clamp(25*mouseX/(cardWidthHalf), -30, 30);

  
  console.log( bound.left, e.clientX)
  // 
  
  card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`
  

  //console.log(mouseX, centerX, cardWidth );
  //e.target.classList.add('shown');
};



// export function Card() {
//   return (<div
//     className="card {types} interactive active loading masked={!!mask}"
//     data-subtypes={subtypes}
//     data-rarity={rarity}
//     style={dynamicStyles}
//   >
//     <div 
//       className="card__translater">
//       <button
//         className="card__rotator"
//         onClick={activate}
//         onPointerMove={interact}
//         onMouseOut={interactEnd}
//         onBlur={deactivate}
//         aria-label="Expand the Pokemon Card; {name}."
//         tabIndex="0"
//         >
//         <Image
//           className="card__back"
//           src={back_img}
//           alt="The back of a Pokemon Card, a Pokeball in the center with Pokemon logo above and below"
//           loading="lazy"
//           width="660"
//           height="921"
//         />
//         <div className="card__front" 
//           style={ staticStyles + foilStyles }>
//           <Image
//             src={front_img}
//             alt="Front design of the {name} Pokemon Card, with the stats and info around the edge"
//             onLoad={imageLoader}
//             loading="lazy"
//             width="660"
//             height="921"
//           />
//           <div className="card__shine"></div>
//           <div className="card__glare"></div>
//         </div>
//       </button>
//     </div>
//   </div>
//   )
// }