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
  <section><img src="ram_design/ramlayout.png"></img></section>
  <li> We will concretely focus on an SRAM memory array with capacity of 16, holding
  4b-wide data elements.</li>
  <li> Inputs: 4-bit address, active high write enable (WE=1 for write, WE=0 for read),
  4-bit data bus, and a single clk signal for synchronization</li>
  <li> Outputs: 4-bit data bus</li>
  <li> A read or write operation must be completed within one clock period</li>
    </ul>
    </section>

  </div>)

  let design = fadeInFactory(<div>
    <h2>1. Design:</h2>
      <section><h3>1.1 Decoder:</h3>
    <p>
    The decoder takes 4 inputs and will output a high on the corresponding wordline
based. The outputs have tristate buffers on them to assure the wordlines will only
output when desired. The enable on the phi_bar allows for the world line to rise
after the new input values are processed.
    </p>
    <figure>
      <img src="ram_design/decoder.png"></img>
      <figcaption>Figure 1.1: Design of the decoder to be used in the selection of the rows</figcaption>
    </figure>
    
    </section>

  </div>)

  let tristate_buffer = fadeInFactory(<div><section>
    <h3>1.2 Tri-state Buffer/Inverter:
    </h3>
    <p>The tri-state buffer and inverter are necessary for controlling the wires within the
    memory in order to create areas of high impedance to isolate the inputs that are desired</p>
    <figure>
     <img src="ram_design/tristate_buffer_1.png"></img> 
     <figcaption>Figure 1.2: Tri-state buffer design to be used in the column driver
     </figcaption>
    </figure>
    <figure>
     <img src="ram_design/tristate_buffer_2.png"></img> 
     <figcaption>Figure 1.3: Tri-state inverter design to be used in the column driver
     </figcaption>
    </figure>
    </section></div>)

    let memory_cell = fadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      
      <figure>
        <img src="ram_design/6t_sram.png" alt="" />
        <figcaption>Figure 1.4: Memory cell design for a single bit using 6T SRAM cell design</figcaption>
      </figure>

      <p>The only transistors that are sized to values other than the minimum are those
concerning the CR and PR, thus the 2 to 1 ratio that is seen on the left hand side iand the
1 to 1 on the right. The values were produced after starting off with the ratios from those
that were present for the technology from the textbook where the ratios should be greater
than 1.2 and less than 1.8, for each side, respectively. Using those as a starting point the
values were tweaked once the single cell was tested with the full bitline capacitance, but
the capacitances from the milestone were too high such that the WL was not able to
charge the capacitors, these values were selected to preserve a similar ratio while assuring
that the WL was able to charge the inputs fast enough</p>
      </section></div>)


      
    let memory_col = fadeInFactory(<div><section>
      <h3>1.4 Memory Column 16 words (1bit):
      </h3>
      <p>The memory column was made as both a testing block and for the sake of clarity.
Taking a single column and placing 4 of cells next to each other will produce the 16
words of length 4 bits.</p>
      <figure>
        <img src="ram_design/mem_col.png" alt="" />
        <figcaption>Figure 1.4: The design of a single column of the memory
        that will be used to make the 4 bit long words</figcaption>
      </figure>
      </section></div>)

      
    let io_bus = fadeInFactory(<div><section>
      <h3>1.5 4-bit IO Bus:
      </h3>
      <p>This 4-bit IO bus is used to interface with the ram driving the inputs and receiving
the outputs. Each input has a gate-based latch in them to increase the robustness of the
design to assure that even if the inputs change while the WE is on the stored output will
be the one that is coinciding with the rising edge of the clock.
</p>
      <figure>
        <img src="ram_design/io_bus.png" alt="" />
        <figcaption>Figure 1.5: The design for the 4-bit IO bus.</figcaption>
      </figure>
      </section></div>)

          
    let full_memory = fadeInFactory(<div><section>
      <h3>1.6 Full Memory (16 words of 4 bits):
      </h3>
      <p>The design of the full memory with IO is layed out into 16 rows and 4 columns to
produce the required 16 words of 4 bits. The decoder takes in a 4 bit input and outputs
high the specific wordline on phi_bar. The bitline is conditioned in the same manner as
the single bit, though there are 4 times as many tri-state buffers to supply all the bitlines,
with a NOR phi_bar and WE to only condition the line for a limited amount of time so as
to not interfere with the desired values be it input or output values. The 4-bit IO bus is
placed at the bottom handling both the input and output values. The values are input and
read on the D0-3 lines.
</p>
      <figure>
        <img src="ram_design/full_ram.png" alt="" />
        <figcaption>Figure 1.6: The design for the full 16 words of 4 bits memory with peripherals
        </figcaption>
      </figure>
      </section></div>)

            
    let validation = fadeInFactory(<div><section>
      <h2>2. Validation:
      </h2>
      <h3>2.1 Decoder:
      </h3>

      <figure>
        <img src="ram_design/val_decoder.png" alt="" />
        <figcaption>Figure 2.1: Test setup for decoder using 4 pulse generator to cover all 16 cases</figcaption>
      </figure>
      <figure>
        <img src="ram_design/val_power.png" alt="" />
        <figcaption>Figure 2.2: Close-up of tht pulse generators each one has double the period of the next
        </figcaption>
      </figure>
      <figure>
        <img src="ram_design/val_decoder_out.png" alt="" />
        <figcaption>Figure 2.2: Plot of the word outputs rising in their designated order
        </figcaption>
      </figure>
      <p>The inputs were test ascending order 0-15 as each wave worldline rises in that it is
      possible to confirm the validity of the decoder</p>
      
      </section></div>)

            
    let validation_tri_state = fadeInFactory(<div><section>
      <h3>2.2 Tri-state Buffer/Inverter:
      </h3>
      <figure>
      <table className="dataframe w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
            <th>EN</th>
            <th>In</th>
            <th>Out</th>
          </tr>
            </thead>
              <tbody>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>0</th>
                  <td>0</td>
                  <td>*</td>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>0</th>
                  <td>1</td>
                  <td>*</td>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>1</th>
                  <td>0</td>
                  <td>0</td>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>1</th>
                  <td>1</td>
                  <td>1</td>
                </tr>
        </tbody>
        </table>
        <figcaption>Figure 2.3: All possible inputs of input and enable for tristate
        </figcaption>
      </figure>
      <figure>
        <img src="ram_design/tristate_val.png" alt="" />
        <figcaption>Figure 2.4: Test setup for the tri-state buffer
        </figcaption>
      </figure>
      <figure>
        <img src="ram_design/tristate_inverter_val.png" alt="" />
        <figcaption>Figure 2.5: Test setup for the tri-state inverter
        </figcaption>
      </figure>
      <figure>
        <img src="ram_design/figure_2_6.png" alt="" />
        <figcaption>Figure 2.6: Plot showing the outputs from the tri-state buffer. When enable is on the output is reflected by the input
        </figcaption>
      </figure>

      <figure>
        <img src="ram_design/figure_2_7.png" alt="" />
        <figcaption>Figure 2.7: Plot showing the outputs from the tri-state inverter When enable is on the output is reflected by the inverted input

        </figcaption>
      </figure>
      
      </section></div>)

            
    let memory_cell_2 = fadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      
      <figure>
        <img src="ram_design/" alt="" />
        <figcaption></figcaption>
      </figure>
      </section></div>)

            
    let memory_cell3 = fadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      
      
      </section></div>)

            
    let memory_cell4 = fadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      
      <figure>
        <img src="ram_design/" alt="" />
        <figcaption></figcaption>
      </figure>
      </section></div>)


      
    let memory_cell5 = fadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      <figure>
        <img src="ram_design/" alt="" />
        <figcaption></figcaption>
      </figure>
      
      </section></div>)


          
    let memory_cell6 = fadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      
      <figure>
        <img src="ram_design/" alt="" />
        <figcaption></figcaption>
      </figure>
      </section></div>)

  return (
      <section className="">
      {design_problem}
      {design}
      {tristate_buffer}
      {memory_cell}
      {memory_col}
      {io_bus}
      {full_memory}
      {validation}
      <br></br>
      {validation_tri_state}
      </section>     
     
    );
}

