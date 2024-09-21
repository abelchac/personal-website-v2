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


function generator_down_table(){
  return (
    <section>
  <table className="dataframe w-full text-left rtl:text-right text-neutral-950 dark:text-neutral-950 text-lg ">
    <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-clay-red dark:border-gray-700">Conv2D (Module 0)</th>
        <td  className="px-2">in=4, out=64, kernel=(3,3), stride=2, padding =1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-clay-red dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2">negative_slope=0.2</td>
      </tr>
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">Conv2D (Module 1)</th>
        <td  className="px-2">in=6, out=128, kernel=(3,3), stride=2, padding =1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">BatchNorm2D </th>
        <td  className="px-2">features=128, momentum=0.8</td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2">negative_slope=0.2</td>
      </tr>
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">Conv2D (Module 2) </th>
        <td  className="px-2" >in=128, out=256, kernel=(3,3), stride=2, padding =1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">BatchNorm2D </th>
        <td  className="px-2" >features=256, momentum=0.8        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2">negative_slope=0.2</td>
      </tr>
      
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">Conv2D (Module 3) </th>
        <td  className="px-2">in=256, out=512, kernel=(3,3), stride=2, padding =1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">BatchNorm2D </th>
        <td  className="px-2">features=512, momentum=0.8        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2">negative_slope=0.2</td>
      </tr>
      
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">Conv2D (Module 4) </th>
        <td  className="px-2">in=512, out=512, kernel=(3,3), stride=2, padding =1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">BatchNorm2D </th>
        <td  className="px-2">features=256, momentum=0.8        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2">negative_slope=0.2</td>
      </tr>
      
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">Conv2D (Module 5) </th>
        <td  className="px-2">in=512, out=512, kernel=(3,3), stride=2, padding =1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">BatchNorm2D </th>
        <td  className="px-2">features=256, momentum=0.8        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2">negative_slope=0.2</td>
      </tr>
      
      </tbody>


      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-blue dark:border-gray-700">Conv2D (Module 6) </th>
        <td  className="px-2">in=512, out=512, kernel=(3,3), stride=2, padding=1, bias=False
        </td> 
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-blue dark:border-gray-700">LeakyReLU </th>
        <td  className="px-2" >negative_slope=0.2</td>
      </tr>
      
      </tbody>
      
    </table>
    </section>
  )
}

function linearLayer(){
  return (  
    <section>
      <table className="dataframe w-full text-left rtl:text-right text-neutral-950 dark:text-neutral-950 text-lg">
      <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Linear</th>
            <td  className="px-2">  in=latent dimension, out=(64 / (2 ^ index of UNetDown Module)) ^ 2
            </td>
          </tr>
          
          </tbody>
          
        </table> </section>    )
}

   
function generator_up_table(){


  return (<section>

<table className="dataframe w-full text-left rtl:text-right text-neutral-950 dark:text-neutral-950 text-lg">
    <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-clay-red dark:border-gray-700">Upsample (Module 0) </th>
        <td  className="px-2">scale_factor=2
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-clay-red dark:border-gray-700">Conv2D </th>
        <td className="px-2">in=512, out=512, kernel=(3,3), stride=1, padding=1, bias=False
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-clay-red dark:border-gray-700">BatchNorm2D </th>
        <td className="px-2">features=512, momentum=0.8</td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-clay-red dark:border-gray-700">ReLU </th>
        <td></td>
      </tr>
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">Upsample (Module 1)</th>
        <td  className="px-2">scale_factor=2
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">Conv2D </th>
        <td className="px-2">in=1024, out=512, kernel=(3,3), stride=1, padding=1, bias=False

        </td>
      </tr>

      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">BatchNorm2D </th>
        <td className="px-2">features=512, momentum=0.8</td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-pink dark:border-gray-700">ReLU </th>
        <td className="px-2"></td>
      </tr>
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">Upsample (Module 2) </th>
        <td  className="px-2">scale_factor=2
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">Conv2D </th>
        <td className="px-2">in=1024, out=512, kernel=(3,3), stride=1, padding=1, bias=False

        </td>
      </tr>

      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">BatchNorm2D </th>
        <td className="px-2">features=512, momentum=0.8       </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-orange dark:border-gray-700">ReLU </th>
        <td className="px-2"></td>
      </tr>
      
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">Upsample (Module 3) </th>
        <td  className="px-2">scale_factor=2
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">Conv2D </th>
        <td className="px-2">in=1024, out=256, kernel=(3,3), stride=1, padding=1, bias=False

        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">BatchNorm2D </th>
        <td className="px-2">features=256, momentum=0.8
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-yellow dark:border-gray-700">ReLU </th>
        <td className="px-2"></td>
      </tr>
      
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">Upsample (Module 4) </th>
        <td  className="px-2">scale_factor=2
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">Conv2D </th>
        <td className="px-2">in=512, out=128, kernel=(3,3), stride=1, padding=1, bias=False

        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">BatchNorm2D </th>
        <td className="px-2">features=128, momentum=0.8        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-light-green dark:border-gray-700">ReLU </th>
        <td className="px-2"></td>
      </tr>
      
      </tbody>

      <tbody>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">Upsample (Module 5) </th>
        <td  className="px-2">scale_factor=2
        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">Conv2D </th>
        <td className="px-2">in=256, out=64, kernel=(3,3), stride=1, padding=1, bias=False

        </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">BatchNorm2D </th>
        <td className="px-2">features=64, momentum=0.8       </td>
      </tr>
      <tr className="bg-white border-b dark:border-gray-700">
        <th className="bg-white border-b dark:bg-green-blue dark:border-gray-700">ReLU </th>
        <td className="px-2"></td>
      </tr>
      
      </tbody>
      
    </table>



  </section>)
}


function processingLayer(){
  return (  
    <section>
      <table className="dataframe w-full text-left rtl:text-right text-neutral-950 dark:text-neutral-950 text-lg">
      <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Upsample</th>
            <td  className="px-2">  scale_factor=2
            </td>
          </tr>

          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Conv2D</th>
            <td  className="px-2"> in=129, out=3 kernel=(3,3), stride=1 padding=1
            </td>
          </tr>

          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Tanh</th>
            <td  className="px-2">  
            </td>
          </tr>
          
          </tbody>
          
        </table> </section>    )
}


function discriminator(){
  return (  
    <section>
      <table className="dataframe w-full text-left rtl:text-right text-neutral-950 dark:text-neutral-950 text-lg">
      <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Conv2D</th>
            <td  className="px-2">  in=3, out=128, kernel=(4,4), stride=2, padding=1, bias=False

            </td>
          </tr>

          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">LeakyReLU </th>
            <td  className="px-2">slope=0.2
            </td>
          </tr>

          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Conv2D</th>
            <td  className="px-2">  in=128, out=256, kernel=(4,4), stride=2, padding=1, bias=False
            </td>
          </tr>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">BatchNorm2D </th>
            <td  className="px-2">  features=256
            </td>
          </tr>
                      
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">LeakyReLU </th>
            <td  className="px-2">slope=0.2
            </td>
          </tr>
          
          
          </tbody>
          

          <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Conv2D</th>
            <td  className="px-2"> in=256, out=512, kernel=(4,4), stride=2, padding=1, bias=False


            </td>
          </tr>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">BatchNorm2D </th>
            <td  className="px-2">  features=512
            </td>
          </tr>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">LeakyReLU </th>
            <td  className="px-2">slope=0.2
            </td>
          </tr>
          
          </tbody>

          <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Conv2D</th>
            <td  className="px-2">in=512, out=1024, kernel=(4,4), stride=2, padding=1, bias=False


            </td>
          </tr>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">BatchNorm2D </th>
            <td  className="px-2">  features=1024

            </td>
          </tr>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">LeakyReLU </th>
            <td  className="px-2">slope=0.2
            </td>
          </tr>
          
          </tbody>


          
          <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Conv2D</th>
            <td  className="px-2">in=1024, out=1, kernel=(4,4), stride=1, padding=0, bias=False


            </td>
          </tr>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Sigmoid </th>
            <td  className="px-2">  </td>
          </tr>
          </tbody>


        </table> </section>    )
}



function quantitative_results(){
  return (  
      <table className="dataframe w-full text-left rtl:text-right text-neutral-950 dark:text-neutral-950 text-lg">
      <tbody>
          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">FID Score</th>
            <td  className="px-2">274.6021487796754
            </td>
          </tr>

          <tr className="border bg-white border-b dark:border-gray-700">
            <th className=" border  bg-white border-b dark:border-gray-700">Average LPIPS 0.17127
</th>
            <td  className="px-2"> Average LPIPS 0.17127

            </td>
          </tr>
          
          </tbody>
          
        </table>   )
}



export default function BicycleGANProject() {
  let title = (<div className=""><h1>BicycleGAN</h1>
    <Link href="https://github.com/abelchac/bicyclegan"> <h2>Project Link: https://github.com/abelchac/bicyclegan</h2></Link>
    </div>);

  
  let video = FadeInFactory(<video controls src="bicyclegan/BicycleGan.mov"></video>);

  let abstract = FadeInFactory(<div>
    <h2>Abstract</h2>
    <p>In the problem of single image-to-image translation, the model learns a single output for each
input image of the edge given, with ambiguous generation. Thus in the many image-to-image
models we must create a method to create multiple outputs on a distribution. We use a generator
that takes in a latent vector that is randomly sampled to develop this distribution. We used both
“add to all” and “and to input” when considering where we include our latent vector.
</p>
  </div>)

let problem = FadeInFactory(<div>
  
  <h2>1 Problem/Introduction</h2>
  <p>The problem that is proposed is to develop a method to generate a many image-to-image model,
meaning that we can create many different realistic outputs based on a single image. Our
approach to the problem is to implement BicycleGAN which is an extension of CycleGAN that
links basic GAN and VAE models in order to map an edge image into multiple RGB images,
which CycleGAN is not capable of.
</p>
<section><img      src="bicyclegan/example.png" alt={""}></img ></section>
</div>)

let lit_review = FadeInFactory(<div>
  <h2>2 Literature Review
  </h2>
  <p>BicycleGAN consists of two major modules, cVAE-GAN and cLR-GAN.
cVAE-GAN encodes a ground truth image B and then attempts to map an image A into B [1].
The Conditional Variational Autoencoder GAN is an approach that generates the latent vector
from the ground truth image. The model should be able to create a specific image from the input.
This method gives the model a “peek” into desired output. The Conditional Latent Regressor
GAN takes a randomly generated latent vector and the image to generate an image that is
realistic, but not necessarily the ground truth image. The point of the BicycleGAN is to combine
these 2 methods to create a connection between the latent vector and the encoding to create a
diverse and conditional set of output images [1].
</p>
</div>)

let exp_sum = FadeInFactory(<div>
  <h2>3 Experiments Summary</h2>
  <p>We began our experiments using the Edges2shoes dataset. This data set consists of paired images
of a single object and its edges, for a total resolution of 512 by 256 pixels. Each object takes up
the entire image. We will first split the paired image into the ground truth RGB image and edge
image, and reduce both to a resolution of 128 by 128 pixels. We will also normalize the pixel
values of the object image to the range [-1 to 1]. The images have white backgrounds. This
should provide good performance for the model, as there is little to no noise and good resolution.
For experimentation we used both the “add to all” and “and to input” methods for the inclusion
of the latent vector. Our findings showed that using the “add to all” method, much like in the
article, gave more noise that allowed for a more normal distribution of possible output images.</p>
</div>)

let genTable = FadeInFactory(generator_down_table());
let linearLayerOut = FadeInFactory(linearLayer());
let generator_up_table_out = FadeInFactory(generator_up_table());
let processingLayerOut = FadeInFactory(processingLayer());
let discrim = FadeInFactory(discriminator());

let method = FadeInFactory
(<div>

  <h2>4 Methodology</h2>
  <p>For our final generator model, we implemented a similar UNet generator used in Linder-Norén’s
Pytorch Bicycle GAN implementation [2].</p>
  <p>Generator (UNetDown Modules):</p>
  {genTable}
  <p>The input of each UNetDown module (denoted by color) is the output of the previous module, or
the image in the case of the first module, concatenated with the latent code after being processed
by a linear layer:
</p>
{linearLayerOut}
  <p>This adds the latent code to every layer of the UNet generator.</p>
  <p>Generator (Up Modules):</p>
  {generator_up_table_out}
  <p>The input of the first UNetUp module (denoted by color) is the output of UNetDown modules 5
and 6 concatenated. The input of every subsequent UNetUp module at layer i is the output of the
previous UNetUp module, and its output is concatenated with the input to the UNetDown
module at layer 6 - i.
</p>
<br></br>
  <p>Finally, we take the output of the UNetUp modules and perform final processing, shown below:</p>
  {processingLayerOut}
  <br></br>
  <p>For our discriminator model we use PatchGAN, shown below:</p>
  {discrim}
  <br></br>
  <p>The output shape of our discriminator is (1, 5x5).</p>
  <br></br>
  <p>For training, we use two separate discriminators and a single generator. For each training image,
We first generate a random latent code, encode it, then generate a prediction from each
discriminator. We calculate loss for the encoder by taking the sum of GAN loss, using BCE
criterion, for each discriminator prediction, L1 pixel loss, and L1 KL divergence loss. We
calculate loss for the generator by taking the L1 loss for latent code reconstruction. We calculate
loss for the discriminators by summing BCE for the valid label with prediction for a true image
and the fake label with prediction for a fake image.</p>
</div>)

let quantitative_results_table = quantitative_results();

let result_images = FadeInFactory(<div>
    <h4>Step 0:</h4>
  <img      src="bicyclegan/step0.png" alt={""}></img >
  <h4>Step 500:</h4>
  <img      src="bicyclegan/step500.png" alt={""}></img >
  <h4>Step 1000:</h4>
  <img      src="bicyclegan/step1000.png" alt={""}></img >
  <h4>Step 4000:</h4>
  <img      src="bicyclegan/step4000.png" alt={""}></img >
  <h4>Step 99000:</h4>
  <img      src="bicyclegan/step99000.png" alt={""}></img >

</div>
);


let results_part2 = FadeInFactory(  <div><h3>Quantitative Evaluations</h3>
  {quantitative_results_table}
  <h3>Qualitative Evaluations</h3>
  <img      src="bicyclegan/qualitative.png" alt={""}></img ></div>)
let results_sum = FadeInFactory(<div>
  <h2>5 Summary Of Results</h2>
  <h3>Evaluation Metrics</h3>
  <h4>Visualization of Training:</h4>
  <div  className="grid grid-cols-[repeat(auto-fill,minmax(6rem,50rem))] gap-x-2 gap-y-2 ">
  <img      src="bicyclegan/total_loss.png" alt={""}></img >
  <img      src="bicyclegan/generator_loss.png" alt={""}></img >
  <img      src="bicyclegan/discrim_loss.png" alt={""}></img >
  <img      src="bicyclegan/discrim_lr_loss.png" alt={""}></img >
  </div>

  {result_images}
  {results_part2}

</div>)

let conclusion = FadeInFactory(<div>
  <h2>Conclusion
  </h2>
  <p>The results that are created by our model are diverse and are conditionally based on our latent
vector. This was aided by the use of the “add to all” method which adds the latent vector to all
layers of the generator instead of just the input. The realistic nature of the images that are
produced could deceive at first glance, but when considering the FID score of over 200, we can
see that these images are not seen as realistic. A possible change that could have been made was
to use a learnable upscale such as a transposed convolution rather than just a raw upscale that has
no learnable features. However, the results produced do give confidence in the validity of the
BicycleGAN model and the use of a latent vector to create bijective mappings of multi-modal
image outputs.
</p>
</div>)

let references = FadeInFactory(<div>
  <h2>References</h2>
  <p>[1] Zhu, J. Y., Zhang, R., Pathak, D., Darrell, T., Efros, A. A., Wang, O., & Shechtman, E. (2017).
Multimodal Image-to-Image Translation by Enforcing Bi-Cycle Consistency. In Advances in neural
information processing systems (pp. 465-476).</p>
<p>[2] https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/bicyclegan</p>
</div>)
 

  return (
      <section className="">
      {title}
      <br></br>
      <section>
      {abstract}
      <br></br>
      {video}
      <br></br>
      </section>
      {problem}
      <br></br>
      {lit_review}
      <br></br>
      {exp_sum}
      <br></br>
      {method}
      <br></br>
      {results_sum}
      <br></br>
      {conclusion}
      <br></br>
      {references}
      
      </section>     
     
    );
}

