
import { useInView, InView} from "react-intersection-observer";

import "../app/globals.css";


export function ProjectCard(props : {project_svg_name : string, alt_text : string, href : string}){
  let data_string = './cards/' + props.project_svg_name;
  
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
  
  let href = "/projects/" + props.href;
  return (
    <InView as="div" onChange={(_, entry) => {entry.target.children[0].classList.add('shown')}} className={`transition-opacity duration-1000 hid zoom ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
      <a href={href}>
      <img ref={ref} className={`place-items-center ProjectCard`} src={data_string} alt={props.alt_text} />
      </a>
    </InView>
  );
}

const activate = (e : any) => {
    
  e.target.classList.add('shown');
};



export function Card() {
  return (<div
    className="card {types} interactive active loading masked={!!mask}"
    data-subtypes={subtypes}
    data-rarity={rarity}
    style={dynamicStyles}
  >
    <div 
      className="card__translater">
      <button
        className="card__rotator"
        onClick={activate}
        onPointerMove={interact}
        onMouseOut={interactEnd}
        onBlur={deactivate}
        aria-label="Expand the Pokemon Card; {name}."
        tabIndex="0"
        >
        <img
          className="card__back"
          src={back_img}
          alt="The back of a Pokemon Card, a Pokeball in the center with Pokemon logo above and below"
          loading="lazy"
          width="660"
          height="921"
        />
        <div className="card__front" 
          style={ staticStyles + foilStyles }>
          <img
            src={front_img}
            alt="Front design of the {name} Pokemon Card, with the stats and info around the edge"
            onLoad={imageLoader}
            loading="lazy"
            width="660"
            height="921"
          />
          <div className="card__shine"></div>
          <div className="card__glare"></div>
        </div>
      </button>
    </div>
  </div>
  )
}