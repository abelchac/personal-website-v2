"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";
import { ProjectCard } from "@/components/Card";

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




export default function Home() {

  //let card_csgo = create_card("CSGO_CARD.svg");

 
  let csgo_card_alt_text = `Cards that looks like a trading card that has the title: 'CSGO WIN Predictor', 
    attributes: Big Data, Machine Learning, Data Visualization. 
    Description: Using feed forward neural networks and CSGO player statistics, a game result was predicted `;

  let bicycle_gan_alt_text = `Implemented BicycleGAN which is an extension of CycleGAN that links basic GAN and VAE models in order to map an edge image into multiple RGB images
  
  
  `


  let cards = [ <ProjectCard project_svg_name="KdTreesCard.png" alt_text="" href="kd_trees"/>,
    <ProjectCard project_svg_name="NbaAwardsCard.png" alt_text="" href="nba_awards_predictor"/>,
    <ProjectCard project_svg_name="BicycleGan.png" alt_text="" href="bicyclegan"/>,
    <ProjectCard project_svg_name="LyricGenAi.png" alt_text="" href="lyric_gen"/>,
    <ProjectCard project_svg_name="NbaShotChartCard.png" alt_text="" href="nba_shot_chart"/>,
    <ProjectCard project_svg_name="RamCard.png" alt_text="" href="ram_design"/> ,
    <ProjectCard project_svg_name="MusicStreamingCard.png" alt_text="" href="music_streaming"/>,
    <ProjectCard project_svg_name="CSGOCARD.png" alt_text={csgo_card_alt_text} href="csgo_predictor"/>,
    <ProjectCard project_svg_name="SnakeAICard.png" alt_text="" href="simple_dqn" />,
    <ProjectCard project_svg_name="AdderCard.png" alt_text="" href="adder" />]

  
    const { ref, inView, entry } = useInView(
      {
        /* Optional config */
        threshold: 0.1,  // Trigger when 10% of the element is in view
        triggerOnce: true,  // Trigger only once
      }
    );
  
    if (inView) {
      entry?.target.classList.add('shown');
    } else{
      entry?.target.classList.remove('shown');
    }
    
    let project_title = fadeInFactory(<div><h1 className="center ProjectTitle">Projects</h1></div>);
  return (
    
  <section>
     
     <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
      <div ref={ref}>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque posuere tristique pretium. Aliquam malesuada laoreet sapien, eu feugiat ligula gravida in. Maecenas porta eu odio eget congue. Nulla turpis nisi, faucibus at vehicula et, sodales sit amet risus. Integer quis turpis porta, pellentesque neque id, tincidunt tellus. Etiam ligula erat, molestie ut laoreet id, hendrerit rhoncus justo. Aliquam dui felis, tristique at augue ac, vestibulum molestie nisi. Sed fermentum, erat quis molestie volutpat, ex sem ultricies magna, vel consequat enim lectus sed massa. Mauris eget justo feugiat dui volutpat vehicula a sit amet purus. Nullam tincidunt condimentum ligula, eget hendrerit nunc. Fusce sed lectus suscipit, dapibus velit convallis, rhoncus enim. Pellentesque vel metus id mi molestie laoreet. Sed sit amet ligula orci. Curabitur scelerisque ex nec felis lobortis hendrerit. Praesent vitae tincidunt metus. Pellentesque eu magna lacinia, cursus orci sed, convallis quam.

        Fusce dictum ac ligula nec accumsan. Praesent non sapien vitae magna vestibulum tempus in vel metus. Nullam tincidunt venenatis felis, ut convallis risus lobortis id. Donec quam massa, sagittis nec tristique ac, tincidunt nec enim. Curabitur congue ultrices varius. Pellentesque pretium nunc non tellus tempor condimentum. Integer vel laoreet ante. Morbi tempor ut lorem ac iaculis. Vivamus ultricies tempor nisi, consectetur ullamcorper ex rutrum et. Curabitur rhoncus mattis odio non tincidunt.

        Maecenas vel ante eu libero sodales ullamcorper nec nec ipsum. Vestibulum felis nulla, mattis eget aliquet non, suscipit non lectus. Maecenas egestas orci non leo ullamcorper volutpat. Donec lacus purus, hendrerit ut dignissim non, posuere id nisi. Quisque sagittis posuere orci quis ullamcorper. Integer turpis metus, bibendum et sagittis in, facilisis vel libero. Morbi ut nisl nibh. Suspendisse vel lacus lorem. Quisque vel leo eu augue porttitor convallis. Aliquam egestas ex at ullamcorper iaculis. Donec ut ante leo. Nam dignissim ante auctor mi suscipit, non auctor lorem volutpat. Nullam dolor nulla, malesuada scelerisque tempor sit amet, blandit nec lacus. Mauris aliquam dolor felis.

        Mauris venenatis tincidunt tincidunt. Vestibulum non massa ut turpis lobortis volutpat. Suspendisse sodales dignissim velit, quis elementum ipsum feugiat tincidunt. Duis quam nunc, gravida ac nisi vel, lacinia consectetur tellus. Donec scelerisque ex nec lorem volutpat interdum. Cras ornare, diam eu rhoncus porta, ante urna tempus sapien, eget bibendum dolor tortor ac ex. Aliquam erat volutpat. Donec vitae dapibus massa. Ut a justo pellentesque, sagittis ex at, rhoncus lacus. Ut ac accumsan arcu. Curabitur aliquet euismod ipsum, ut ultricies mi efficitur posuere. Praesent fermentum tincidunt erat in cursus. Ut vestibulum lobortis consequat. Vestibulum dignissim nunc nec orci fringilla, et tempus ipsum consequat. Curabitur a tincidunt purus, eu euismod leo.

        Vestibulum fermentum id dui ac interdum. Vivamus vel hendrerit sapien, nec maximus lorem. Nam pellentesque ullamcorper nunc. Aliquam aliquam tortor sed justo fermentum, a dignissim leo sollicitudin. Phasellus in nunc ultrices mi aliquet consectetur ac vitae mauris. Donec eu sollicitudin dui, et aliquam felis. Morbi semper at massa vel vestibulum. Nam sagittis pellentesque nunc ut dapibus. Nulla cursus, orci at mattis posuere, sem leo scelerisque mi, quis scelerisque mi diam eu eros.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque posuere tristique pretium. Aliquam malesuada laoreet sapien, eu feugiat ligula gravida in. Maecenas porta eu odio eget congue. Nulla turpis nisi, faucibus at vehicula et, sodales sit amet risus. Integer quis turpis porta, pellentesque neque id, tincidunt tellus. Etiam ligula erat, molestie ut laoreet id, hendrerit rhoncus justo. Aliquam dui felis, tristique at augue ac, vestibulum molestie nisi. Sed fermentum, erat quis molestie volutpat, ex sem ultricies magna, vel consequat enim lectus sed massa. Mauris eget justo feugiat dui volutpat vehicula a sit amet purus. Nullam tincidunt condimentum ligula, eget hendrerit nunc. Fusce sed lectus suscipit, dapibus velit convallis, rhoncus enim. Pellentesque vel metus id mi molestie laoreet. Sed sit amet ligula orci. Curabitur scelerisque ex nec felis lobortis hendrerit. Praesent vitae tincidunt metus. Pellentesque eu magna lacinia, cursus orci sed, convallis quam.

        Fusce dictum ac ligula nec accumsan. Praesent non sapien vitae magna vestibulum tempus in vel metus. Nullam tincidunt venenatis felis, ut convallis risus lobortis id. Donec quam massa, sagittis nec tristique ac, tincidunt nec enim. Curabitur congue ultrices varius. Pellentesque pretium nunc non tellus tempor condimentum. Integer vel laoreet ante. Morbi tempor ut lorem ac iaculis. Vivamus ultricies tempor nisi, consectetur ullamcorper ex rutrum et. Curabitur rhoncus mattis odio non tincidunt.

        Maecenas vel ante eu libero sodales ullamcorper nec nec ipsum. Vestibulum felis nulla, mattis eget aliquet non, suscipit non lectus. Maecenas egestas orci non leo ullamcorper volutpat. Donec lacus purus, hendrerit ut dignissim non, posuere id nisi. Quisque sagittis posuere orci quis ullamcorper. Integer turpis metus, bibendum et sagittis in, facilisis vel libero. Morbi ut nisl nibh. Suspendisse vel lacus lorem. Quisque vel leo eu augue porttitor convallis. Aliquam egestas ex at ullamcorper iaculis. Donec ut ante leo. Nam dignissim ante auctor mi suscipit, non auctor lorem volutpat. Nullam dolor nulla, malesuada scelerisque tempor sit amet, blandit nec lacus. Mauris aliquam dolor felis.

        Mauris venenatis tincidunt tincidunt. Vestibulum non massa ut turpis lobortis volutpat. Suspendisse sodales dignissim velit, quis elementum ipsum feugiat tincidunt. Duis quam nunc, gravida ac nisi vel, lacinia consectetur tellus. Donec scelerisque ex nec lorem volutpat interdum. Cras ornare, diam eu rhoncus porta, ante urna tempus sapien, eget bibendum dolor tortor ac ex. Aliquam erat volutpat. Donec vitae dapibus massa. Ut a justo pellentesque, sagittis ex at, rhoncus lacus. Ut ac accumsan arcu. Curabitur aliquet euismod ipsum, ut ultricies mi efficitur posuere. Praesent fermentum tincidunt erat in cursus. Ut vestibulum lobortis consequat. Vestibulum dignissim nunc nec orci fringilla, et tempus ipsum consequat. Curabitur a tincidunt purus, eu euismod leo.

        Vestibulum fermentum id dui ac interdum. Vivamus vel hendrerit sapien, nec maximus lorem. Nam pellentesque ullamcorper nunc. Aliquam aliquam tortor sed justo fermentum, a dignissim leo sollicitudin. Phasellus in nunc ultrices mi aliquet consectetur ac vitae mauris. Donec eu sollicitudin dui, et aliquam felis. Morbi semper at massa vel vestibulum. Nam sagittis pellentesque nunc ut dapibus. Nulla cursus, orci at mattis posuere, sem leo scelerisque mi, quis scelerisque mi diam eu eros.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque posuere tristique pretium. Aliquam malesuada laoreet sapien, eu feugiat ligula gravida in. Maecenas porta eu odio eget congue. Nulla turpis nisi, faucibus at vehicula et, sodales sit amet risus. Integer quis turpis porta, pellentesque neque id, tincidunt tellus. Etiam ligula erat, molestie ut laoreet id, hendrerit rhoncus justo. Aliquam dui felis, tristique at augue ac, vestibulum molestie nisi. Sed fermentum, erat quis molestie volutpat, ex sem ultricies magna, vel consequat enim lectus sed massa. Mauris eget justo feugiat dui volutpat vehicula a sit amet purus. Nullam tincidunt condimentum ligula, eget hendrerit nunc. Fusce sed lectus suscipit, dapibus velit convallis, rhoncus enim. Pellentesque vel metus id mi molestie laoreet. Sed sit amet ligula orci. Curabitur scelerisque ex nec felis lobortis hendrerit. Praesent vitae tincidunt metus. Pellentesque eu magna lacinia, cursus orci sed, convallis quam.

        Fusce dictum ac ligula nec accumsan. Praesent non sapien vitae magna vestibulum tempus in vel metus. Nullam tincidunt venenatis felis, ut convallis risus lobortis id. Donec quam massa, sagittis nec tristique ac, tincidunt nec enim. Curabitur congue ultrices varius. Pellentesque pretium nunc non tellus tempor condimentum. Integer vel laoreet ante. Morbi tempor ut lorem ac iaculis. Vivamus ultricies tempor nisi, consectetur ullamcorper ex rutrum et. Curabitur rhoncus mattis odio non tincidunt.

        Maecenas vel ante eu libero sodales ullamcorper nec nec ipsum. Vestibulum felis nulla, mattis eget aliquet non, suscipit non lectus. Maecenas egestas orci non leo ullamcorper volutpat. Donec lacus purus, hendrerit ut dignissim non, posuere id nisi. Quisque sagittis posuere orci quis ullamcorper. Integer turpis metus, bibendum et sagittis in, facilisis vel libero. Morbi ut nisl nibh. Suspendisse vel lacus lorem. Quisque vel leo eu augue porttitor convallis. Aliquam egestas ex at ullamcorper iaculis. Donec ut ante leo. Nam dignissim ante auctor mi suscipit, non auctor lorem volutpat. Nullam dolor nulla, malesuada scelerisque tempor sit amet, blandit nec lacus. Mauris aliquam dolor felis.

        Mauris venenatis tincidunt tincidunt. Vestibulum non massa ut turpis lobortis volutpat. Suspendisse sodales dignissim velit, quis elementum ipsum feugiat tincidunt. Duis quam nunc, gravida ac nisi vel, lacinia consectetur tellus. Donec scelerisque ex nec lorem volutpat interdum. Cras ornare, diam eu rhoncus porta, ante urna tempus sapien, eget bibendum dolor tortor ac ex. Aliquam erat volutpat. Donec vitae dapibus massa. Ut a justo pellentesque, sagittis ex at, rhoncus lacus. Ut ac accumsan arcu. Curabitur aliquet euismod ipsum, ut ultricies mi efficitur posuere. Praesent fermentum tincidunt erat in cursus. Ut vestibulum lobortis consequat. Vestibulum dignissim nunc nec orci fringilla, et tempus ipsum consequat. Curabitur a tincidunt purus, eu euismod leo.

        Vestibulum fermentum id dui ac interdum. Vivamus vel hendrerit sapien, nec maximus lorem. Nam pellentesque ullamcorper nunc. Aliquam aliquam tortor sed justo fermentum, a dignissim leo sollicitudin. Phasellus in nunc ultrices mi aliquet consectetur ac vitae mauris. Donec eu sollicitudin dui, et aliquam felis. Morbi semper at massa vel vestibulum. Nam sagittis pellentesque nunc ut dapibus. Nulla cursus, orci at mattis posuere, sem leo scelerisque mi, quis scelerisque mi diam eu eros.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque posuere tristique pretium. Aliquam malesuada laoreet sapien, eu feugiat ligula gravida in. Maecenas porta eu odio eget congue. Nulla turpis nisi, faucibus at vehicula et, sodales sit amet risus. Integer quis turpis porta, pellentesque neque id, tincidunt tellus. Etiam ligula erat, molestie ut laoreet id, hendrerit rhoncus justo. Aliquam dui felis, tristique at augue ac, vestibulum molestie nisi. Sed fermentum, erat quis molestie volutpat, ex sem ultricies magna, vel consequat enim lectus sed massa. Mauris eget justo feugiat dui volutpat vehicula a sit amet purus. Nullam tincidunt condimentum ligula, eget hendrerit nunc. Fusce sed lectus suscipit, dapibus velit convallis, rhoncus enim. Pellentesque vel metus id mi molestie laoreet. Sed sit amet ligula orci. Curabitur scelerisque ex nec felis lobortis hendrerit. Praesent vitae tincidunt metus. Pellentesque eu magna lacinia, cursus orci sed, convallis quam.

        Fusce dictum ac ligula nec accumsan. Praesent non sapien vitae magna vestibulum tempus in vel metus. Nullam tincidunt venenatis felis, ut convallis risus lobortis id. Donec quam massa, sagittis nec tristique ac, tincidunt nec enim. Curabitur congue ultrices varius. Pellentesque pretium nunc non tellus tempor condimentum. Integer vel laoreet ante. Morbi tempor ut lorem ac iaculis. Vivamus ultricies tempor nisi, consectetur ullamcorper ex rutrum et. Curabitur rhoncus mattis odio non tincidunt.

        Maecenas vel ante eu libero sodales ullamcorper nec nec ipsum. Vestibulum felis nulla, mattis eget aliquet non, suscipit non lectus. Maecenas egestas orci non leo ullamcorper volutpat. Donec lacus purus, hendrerit ut dignissim non, posuere id nisi. Quisque sagittis posuere orci quis ullamcorper. Integer turpis metus, bibendum et sagittis in, facilisis vel libero. Morbi ut nisl nibh. Suspendisse vel lacus lorem. Quisque vel leo eu augue porttitor convallis. Aliquam egestas ex at ullamcorper iaculis. Donec ut ante leo. Nam dignissim ante auctor mi suscipit, non auctor lorem volutpat. Nullam dolor nulla, malesuada scelerisque tempor sit amet, blandit nec lacus. Mauris aliquam dolor felis.

        Mauris venenatis tincidunt tincidunt. Vestibulum non massa ut turpis lobortis volutpat. Suspendisse sodales dignissim velit, quis elementum ipsum feugiat tincidunt. Duis quam nunc, gravida ac nisi vel, lacinia consectetur tellus. Donec scelerisque ex nec lorem volutpat interdum. Cras ornare, diam eu rhoncus porta, ante urna tempus sapien, eget bibendum dolor tortor ac ex. Aliquam erat volutpat. Donec vitae dapibus massa. Ut a justo pellentesque, sagittis ex at, rhoncus lacus. Ut ac accumsan arcu. Curabitur aliquet euismod ipsum, ut ultricies mi efficitur posuere. Praesent fermentum tincidunt erat in cursus. Ut vestibulum lobortis consequat. Vestibulum dignissim nunc nec orci fringilla, et tempus ipsum consequat. Curabitur a tincidunt purus, eu euismod leo.

        Vestibulum fermentum id dui ac interdum. Vivamus vel hendrerit sapien, nec maximus lorem. Nam pellentesque ullamcorper nunc. Aliquam aliquam tortor sed justo fermentum, a dignissim leo sollicitudin. Phasellus in nunc ultrices mi aliquet consectetur ac vitae mauris. Donec eu sollicitudin dui, et aliquam felis. Morbi semper at massa vel vestibulum. Nam sagittis pellentesque nunc ut dapibus. Nulla cursus, orci at mattis posuere, sem leo scelerisque mi, quis scelerisque mi diam eu eros.
        </div>    
        </InView>
    <section>
    <br></br>
    {project_title}
    <div className="projectCardGrid grid grid-cols-[repeat(auto-fill,minmax(6rem,25rem))] gap-x-16 gap-y-10 justify-center" >
    {cards[0]}
    {cards[1]}
    {cards[2]}
    {cards[3]}
    {cards[4]}
    {cards[5]}
    {cards[6]}
    {cards[7]}
    {cards[8]}
    {cards[9]}

    </div>  
    </section>   
  </section> 
    
  );
}

