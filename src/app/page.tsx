import Image from "next/image";


function ProjectCard(props : {project_svg_name : string}){
  let data_string = './cards/' + props.project_svg_name;
  console.log(data_string);
  return (<img src="./cards/CSGO_CARD.png"/>);
}


export default function Home() {

  //let card_csgo = create_card("CSGO_CARD.svg");
  return (
    
  <span>

    <ProjectCard project_svg_name="CSGO_CARD.svg"/>
    <ProjectCard project_svg_name="CSGO_CARD.svg"/>
    <ProjectCard project_svg_name="CSGO_CARD.svg"/>
    <ProjectCard project_svg_name="CSGO_CARD.svg"/>
    <ProjectCard project_svg_name="CSGO_CARD.svg"/>
    
 
  </span> 
    
  );
}

