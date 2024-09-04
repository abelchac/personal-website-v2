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



export default function BicycleGANProject() {
  let title = (<div className=""><h1>BicycleGAN</h1>
    <Link href="https://github.com/abelchac/bicyclegan"> <h2>Project Link: https://github.com/abelchac/bicyclegan</h2></Link>
    </div>);

  let abstract = fadeInFactory(<div>
    <h2>Abstract</h2>
    <p>In the problem of single image-to-image translation, the model learns a single output for each
input image of the edge given, with ambiguous generation. Thus in the many image-to-image
models we must create a method to create multiple outputs on a distribution. We use a generator
that takes in a latent vector that is randomly sampled to develop this distribution. We used both
“add to all” and “and to input” when considering where we include our latent vector.
</p>
    <img src="bicyclegan/"></img>
  </div>)

let problem = fadeInFactory(<div>
  <h2>1 Problem/Introduction</h2>
  <p>The problem that is proposed is to develop a method to generate a many image-to-image model,
meaning that we can create many different realistic outputs based on a single image. Our
approach to the problem is to implement BicycleGAN which is an extension of CycleGAN that
links basic GAN and VAE models in order to map an edge image into multiple RGB images,
which CycleGAN is not capable of.
</p>
  <img src="bicyclegan/"></img>
</div>)

let lit_review = fadeInFactory(<div>
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
  <img src="bicyclegan/"></img>
</div>)

let exp_sum = fadeInFactory(<div>
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
  <img src="bicyclegan/"></img>
</div>)

let method = fadeInFactory(<div>
  <h2>4 Methodology</h2>
  <p>For our final generator model, we implemented a similar UNet generator used in Linder-Norén’s
Pytorch Bicycle GAN implementation [2].</p>
  <p>Generator (UNetDown Modules):</p>
  <p>The input of each UNetDown module (denoted by color) is the output of the previous module, or
the image in the case of the first module, concatenated with the latent code after being processed
by a linear layer:
</p>
  <p>This adds the latent code to every layer of the UNet generator.</p>
  <p>Generator (Up Modules):</p>
  <p>The input of the first UNetUp module (denoted by color) is the output of UNetDown modules 5
and 6 concatenated. The input of every subsequent UNetUp module at layer i is the output of the
previous UNetUp module, and its output is concatenated with the input to the UNetDown
module at layer 6 - i.
</p>
  <p>Finally, we take the output of the UNetUp modules and perform final processing, shown below:</p>
  <p>For our discriminator model we use PatchGAN, shown below:</p>
  <p>The output shape of our discriminator is (1, 5x5).</p>
  <p>For training, we use two separate discriminators and a single generator. For each training image,
We first generate a random latent code, encode it, then generate a prediction from each
discriminator. We calculate loss for the encoder by taking the sum of GAN loss, using BCE
criterion, for each discriminator prediction, L1 pixel loss, and L1 KL divergence loss. We
calculate loss for the generator by taking the L1 loss for latent code reconstruction. We calculate
loss for the discriminators by summing BCE for the valid label with prediction for a true image
and the fake label with prediction for a fake image.</p>
  <img src="bicyclegan/"></img>
</div>)

let results_sum = fadeInFactory(<div>
  <h2>5 Summary Of Results</h2>
  <h3>Evaluation Metrics</h3>
  <h4>Visualization of Training:</h4>
  <img src="bicyclegan/"></img>
  <img src="bicyclegan/"></img>
  <img src="bicyclegan/"></img>
  <img src="bicyclegan/"></img>
  <h4>Step 0:</h4>
  <img src="bicyclegan/"></img>
  <h4>Step 500:</h4>
  <img src="bicyclegan/"></img>
  <h4>Step 1000:</h4>
  <img src="bicyclegan/"></img>
  <h4>Step 4000:</h4>
  <img src="bicyclegan/"></img>
  <h4>Step 99000:</h4>
  <img src="bicyclegan/"></img>
  <h3>Quantitative Evaluations</h3>
  <h3>Qualitative Evaluations</h3>
  <img src="bicyclegan/"></img>
</div>)

let conclusion = fadeInFactory(<div>
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

let references = fadeInFactory(<div>
  <h2>References</h2>
  <p>[1] Zhu, J. Y., Zhang, R., Pathak, D., Darrell, T., Efros, A. A., Wang, O., & Shechtman, E. (2017).
Multimodal Image-to-Image Translation by Enforcing Bi-Cycle Consistency. In Advances in neural
information processing systems (pp. 465-476).</p>
<p>[2] https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/bicyclegan</p>
</div>)
 

  return (
      <section className="">
      {title}
      <section>
      {abstract}
      </section>
      {problem}
      {lit_review}
      {exp_sum}
      {method}
      {results_sum}
      {conclusion}
      {references}
      
      </section>     
     
    );
}

