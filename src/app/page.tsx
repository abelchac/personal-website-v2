import Image from "next/image";


function ProjectCard(props : {project_svg_name : string, alt_text : string}){
  let data_string = './cards/' + props.project_svg_name;
  console.log(data_string);
  return (<img className="place-items-center" src={data_string} alt={props.alt_text}/>
  );
}


export default function Home() {

  //let card_csgo = create_card("CSGO_CARD.svg");
  return (
    
  <span>
  <div>
    <div className="grid grid-cols-[repeat(auto-fill,20rem)] gap-x-16 gap-y-10 border-2 p-8 justify-center">
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

