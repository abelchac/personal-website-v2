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
  <section><Image loading="eager" className="!relative" fill  src="/ram_design/ramlayout.png" alt={""}></Image></section>
  <li> We will concretely focus on an SRAM memory array with capacity of 16, holding
  4b-wide data elements.</li>
  <li> Inputs: 4-bit address, active high write enable (WE=1 for write, WE=0 for read),
  4-bit data bus, and a single clk signal for synchronization</li>
  <li> Outputs: 4-bit data bus</li>
  <li> A read or write operation must be completed within one clock period</li>
    </ul>
    </section>

  </div>)

  let design = FadeInFactory(<div>
    <h2>1. Design:</h2>
      <section><h3>1.1 Decoder:</h3>
    <p>
    The decoder takes 4 inputs and will output a high on the corresponding wordline
based. The outputs have tristate buffers on them to assure the wordlines will only
output when desired. The enable on the phi_bar allows for the world line to rise
after the new input values are processed.
    </p>
    <figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/decoder.png" alt={""}></Image>
      <figcaption>Figure 1.1: Design of the decoder to be used in the selection of the rows</figcaption>
    </figure>
    
    </section>

  </div>)

  let tristate_buffer = FadeInFactory(<div><section>
    <h3>1.2 Tri-state Buffer/Inverter:
    </h3>
    <p>The tri-state buffer and inverter are necessary for controlling the wires within the
    memory in order to create areas of high impedance to isolate the inputs that are desired</p>
    <figure>
     <Image loading="eager" className="!relative" fill  src="/ram_design/tristate_buffer_1.png" alt={""}></Image> 
     <figcaption>Figure 1.2: Tri-state buffer design to be used in the column driver
     </figcaption>
    </figure>
    <figure>
     <Image loading="eager" className="!relative" fill  src="/ram_design/tristate_buffer_2.png" alt={""}></Image> 
     <figcaption>Figure 1.3: Tri-state inverter design to be used in the column driver
     </figcaption>
    </figure>
    </section></div>)

    let memory_cell = FadeInFactory(<div><section>
      <h3>1.3 Memory Cell:
      </h3>
      
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/6t_sram.png" alt="" />
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


      
    let memory_col = FadeInFactory(<div><section>
      <h3>1.4 Memory Column 16 words (1bit):
      </h3>
      <p>The memory column was made as both a testing block and for the sake of clarity.
Taking a single column and placing 4 of cells next to each other will produce the 16
words of length 4 bits.</p>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/mem_col.png" alt="" />
        <figcaption>Figure 1.4: The design of a single column of the memory
        that will be used to make the 4 bit long words</figcaption>
      </figure>
      </section></div>)

      
    let io_bus = FadeInFactory(<div><section>
      <h3>1.5 4-bit IO Bus:
      </h3>
      <p>This 4-bit IO bus is used to interface with the ram driving the inputs and receiving
the outputs. Each input has a gate-based latch in them to increase the robustness of the
design to assure that even if the inputs change while the WE is on the stored output will
be the one that is coinciding with the rising edge of the clock.
</p>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/io_bus.png" alt="" />
        <figcaption>Figure 1.5: The design for the 4-bit IO bus.</figcaption>
      </figure>
      </section></div>)

          
    let full_memory = FadeInFactory(<div><section>
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
        <Image loading="eager" className="!relative" fill  src="/ram_design/full_ram.png" alt="" />
        <figcaption>Figure 1.6: The design for the full 16 words of 4 bits memory with peripherals
        </figcaption>
      </figure>
      </section></div>)

            
    let validation = FadeInFactory(<div>
            <h2>2. Validation:
            </h2>
      <section>

      <h3>2.1 Decoder:
      </h3>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/val_decoder.png" alt="" />
        <figcaption>Figure 2.1: Test setup for decoder using 4 pulse generator to cover all 16 cases</figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/val_power.png" alt="" />
        <figcaption>Figure 2.2: Close-up of tht pulse generators each one has double the period of the next
        </figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/val_decoder_out.png" alt="" />
        <figcaption>Figure 2.2: Plot of the word outputs rising in their designated order
        </figcaption>
      </figure>
      <p>The inputs were test ascending order 0-15 as each wave worldline rises in that it is
      possible to confirm the validity of the decoder</p>
      
      </section></div>)

            
    let validation_tri_state = FadeInFactory(<div><section>
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
        <Image loading="eager" className="!relative" fill  src="/ram_design/tristate_val.png" alt="" />
        <figcaption>Figure 2.4: Test setup for the tri-state buffer
        </figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/tristate_inverter_val.png" alt="" />
        <figcaption>Figure 2.5: Test setup for the tri-state inverter
        </figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_6.png" alt="" />
        <figcaption>Figure 2.6: Plot showing the outputs from the tri-state buffer. When enable is on the output is reflected by the input
        </figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_7.png" alt="" />
        <figcaption>Figure 2.7: Plot showing the outputs from the tri-state inverter When enable is on the output is reflected by the inverted input

        </figcaption>
      </figure>
      
      </section></div>)

            
    let memory_cell_val = FadeInFactory(<div><section>
      <h3>2.3 Memory Cell:

      </h3>
      <p>Testing operation: W0 R0 R0 W1 R1 R1 W0 R0 W1 R1 W1 R1 W0 R0</p>

      <p>
The operations that are tested are first writing 0 then read twice to see if the write
is held within the cell. The same process is repeated with writing 1. Then after it
will write a 0 and 1 to see if values can be overwritten with only a single read for
the sake of parsing. The same process for 0 then 1. Both values are read out after
the overwrite.
</p>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_8.png" alt="" />
        <figcaption>Figure 2.8: Design of single bit memory test.</figcaption>
      </figure>

      <p>The testing of the 1 bit memory cell is conducted through 2 PWL generators for the input
and the write enable. Furthermore the WL is kept from being active at all times through a
tri-state buffer as the WL will be brought high on the phi_bar high so that either bitline and
bitline bar can be set high or their respective values can be driven before WL is raised. The latch
is used into the input of the column driver to make sure that if values are changed within write
enable only what was input on the rising edge will be stored. The two phase clock running at
500mhz enables this functionality. The bitline condition is conducted through the use of the
NOR gate on the left side which NORs phi and WE, such that the only time the bitlines are both
conditioned is when the memory is being read, but not for the whole duration of the clock cycle.
The 2 transistors emulate the bitline capacitance that would be seen in the full 16 word memory.
</p>

    <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_9.png" alt="" />
        <figcaption>Figure 2.9:Annotated Plot of in, out, RE, and WE</figcaption>
      </figure>

      <div  className="grid grid-cols-2 gap-x-2 gap-y-2 ">
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_10.png" alt="" />
        <figcaption>Figure 2.10: Plot of WE</figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_11.png" alt="" />
        <figcaption>Figure 2.11: Plot of In</figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_12.png" alt="" />
        <figcaption>Figure 2.12: Plot of out and RE.</figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_13.png" alt="" />
        <figcaption>Figure 2.13 Plot of out</figcaption>
      </figure>
      </div>
      <br></br>
      <p>Whenever the RE is high is when there is a reading operation this is just the inverter WE
(write enable).
On the first write the input is 0 and the value of the output drops to 0 when the RE swings
high. And the 0 is sustained on another read. The same is true once a 1 is written at 8ns.
Then, starting at about 14ps, a 0 is written then a 1 is written to read 1. Finally, around
22ps, a 1 is written, a 0 is written and then a 0 is read covering all the cases that were laid
out above with correctness.</p>
      </section></div>)
    
            
    let mem_col_val = FadeInFactory(<div><section>
      <h3>2.4 Memory Column 16 words (1bit):
      </h3>
      <p>The testing of the column memory was conducted on 2 of memory cells within
the column</p>
<figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_14.png" alt="" />
        <figcaption>Figure 2.14: The design for a 16 word (1 bit) testing        </figcaption>
      </figure>
      
      </section></div>)

            
    let mem_row_val = FadeInFactory(<div><section>
      <h3>2.5 Memory Row 1 Word (4bit):
      </h3>
      <p>The memory row design was solely made for testing purposes to determine if a
single row could be read and written to. The 4 bit lines are conditions on the same NOR
logic as the previous tests. All the bitlines are also loaded with the calculated bitline
capacitance to represent the rest of the memory cells on the bitline. This is the first case
of testing with the 4-bit IO bus to drive the inputs and read the outputs. The test case that
was committed to the row was to read and write 0 to 15 in binary to cover all entries that
could be stored within the single word. Observing the 4 plots of d0-3 it is possible to
observe each output rising in ascending order for counting from 0 to 15 in binary as d3 as
the most significant bit.</p>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_15.png" alt="" />
        <figcaption>Figure 2.15: The design for a 1 word (4 bit) testing
        </figcaption>
      </figure>
      <br></br>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_16.png" alt="" />
        <figcaption>Figure 2.16: Plot of d1-3 and WE</figcaption>
      </figure>

      <br></br>
      <div  className="grid grid-cols-2 gap-x-2 gap-y-2 ">
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_17.png" alt="" />
        <figcaption>Figure 2.17: Plot of d0</figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_18.png" alt="" />
        <figcaption>Figure 2.18: Plot of d1</figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_19.png" alt="" />
        <figcaption>Figure 2.19: Plot of d2.</figcaption>
      </figure>

      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_20.png" alt="" />
        <figcaption>Figure 2.20 Plot of d3</figcaption>
      </figure>
      </div>
      <br></br>
      </section></div>)


    let figure_2_21 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_21.png" alt="" />
      <figcaption>Figure 2.21: The design for a 16 word (4 bit) testing</figcaption>
    </figure>);

    let figure_2_22 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_22.png" alt="" />
      <figcaption>Figure 2.22: Testing a single bit at the 1Ghz frequency to ensure operation</figcaption>
    </figure>);

    let figure_2_23 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_23.png" alt="" />
      <figcaption>Figure 2.23: The pulse generator for controlling the decoder inputs giving 0-15 in 16ns</figcaption>
    </figure>);
    let figure_2_24 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_24.png" alt="" />
      <figcaption>Figure 2.24: The inputs to the memory with pulse generator inputs giving 0-15 in 16n</figcaption>
    </figure>);

    let figure_2_25 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_25.png" alt="" />
      <figcaption>Figure 2.25: Pulse generator for the WE writing for the first 16ns then reading for the next 16ns.</figcaption>
    </figure>);
    
    let figure_2_26 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_26.png" alt="" />
      <figcaption>Figure 2.26: plot of WE, d0-3 (parsed in following figures)
      </figcaption>
    </figure>);
    let figure_2_27 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_27.png" alt="" />
      <figcaption>Figure 2.27: Plot of the decoder input values ranging from 0-15</figcaption>
    </figure>);
    let figure_2_28 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_28.png" alt="" />
      <figcaption>Figure 2.28: The plot of WE which is high for the first 16ns then low for 16ns</figcaption>
    </figure>);
    let figure_2_29 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_29.png" alt="" />
      <figcaption>Figure 2.29: Plot of the input into writing memory values ranging from 0-15</figcaption>
    </figure>);
    let figure_2_30 = FadeInFactory(<figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/figure_2_30.png" alt="" />
      <figcaption>Figure 2.30: Plot of the out of the reading memory values ranging from 0-15</figcaption>
    </figure>);

    let full_mem_val = FadeInFactory(<div><section>
      <h3>2.6 Full Memory (16 words of 4 bits):
      </h3>
      <p>The first objective was just to increase the frequency of the clock, from 500Mhz, to stress test the
memory, the memory was able to run 1Ghz without failure.
Testing operation(1 cell in memory): W0 R0 R0 W1 R1 R1 W0 R0 W1 R1 W1 R1 W0 R0</p>


      {figure_2_21}
      {figure_2_22}
      <p>The full test of the memory will be with writing 0-15 in binary to each row then reading all the
      values back out to see if memory cells have stored the values.</p>
      {figure_2_23}
      {figure_2_24}
      {figure_2_25}
      {figure_2_26}
      {figure_2_27}
      <p>The decoder repeats the same outputs/inputs after the first 16ns</p>

      {figure_2_28}
      {figure_2_29}
      {figure_2_30}
      <p>The time from 16-32ns is the reading operation which reads out 0 to 15 from each of the 16 words. 
        These values are read cannot be from any other source than the memory cells as the input values are cut off with a tri-state buffer when WE is off.
         The decoder is only connected to the WL, such that the values produced must be stored and read from the 16 words of length 4.</p>
      
      
      </section></div>)


          
    let design_metrics = FadeInFactory(<div>
      <h2>3. Design Metrics:</h2>
      <section>
      <h3>3.1 Memory Cell Area:

      </h3>

      <p>Sum of widths in 6T Cell = 7 </p>
      <p>Area = 16 words * 4 bits * Sum of widths in 6T Cell * 22nm = 9856 * 10<sup>-9</sup></p>

      </section></div>)


    let delay = FadeInFactory(<div><section>
      <h3>3.2 Delay:
      </h3>
      <p>Operation to find critical path of read and write: W0 R0 W1 R1
      By writing and reading both a 0 and 1, we can determine the components which
      are a part of the critical path, as the design is symmetric, writing to and reading
      from any word line is the same.
      </p>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_3_1.png" alt="" />
        <figcaption>Figure 3.1: Plot of all changing values that affect memory</figcaption>
      </figure>
      <p>By observing the waveforms it, the bitline (yellow) is the value that is constantly the last
      waveform to settle to the desired value. As in a write bitline should settle to the value that
      is being input and in a read bitline should settle to the stored value. Now by taking the
      time difference between the WE either being high or low and the settled value of the
      bitline the delay can be found.
      </p>
      <p>Write 1 delay: 3.13131e-10s      </p>
      <p>Read 0 delay: 2.96875e-10s
      </p>

      </section></div>)

    let power = FadeInFactory(<div><section>
      <h3>3.3 Power:
      </h3>
      <p>The decoder is given inputs much like the decoder test in section 2.1. The A0 bit
      pulses with a period of 1 and the each next bit is twice as the previous. This will select all
      words in a 16ns span which is as quickly as possible with the clock cycle being 1ns. The
      inputs D0-3 are fed with the same input which has a period of 32ns and pulse width of
      16ns which will give a solid 1 or 0 for the whole duration of selection of rows</p>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_3_1.png" alt="" />
        <figcaption>Figure 3.1: Plot of all changing values that affect memory</figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_3_2.png" alt="" />
        <figcaption>Figure 3.2: The voltage pulse that are inputs into the decoder</figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_3_3.png" alt="" />
        <figcaption>Figure 3.3: The voltage pulse that drives the input into the memory
        </figcaption>
      </figure>
      <figure>
        <Image loading="eager" className="!relative" fill  src="/ram_design/figure_3_4.png" alt="" />
        <figcaption>Figure 3.4: The plot of the voltage that drives the input into the memory
        </figcaption>
      </figure>
      <Image loading="eager" className="!relative" fill  src="/ram_design/yint.png" alt="" />
      <p>The integral value of the power consumed for filling of 0s and 1s into the memory array.
      </p>
      <p>Total power expended: 6.50962 * 10<sup>-12</sup>W</p>
      <p>Average Power: 6.50962 * 10<sup>-12</sup>J / 4 = 1.627405* 10<sup>-12</sup>W</p>
      
     
     

      </section></div>)
          
    let fom = FadeInFactory(<div><section>
      <br></br>
      <h3> 3.4 FOM:
      </h3>
      <p> FOM = 60 * Memory Cell Area * Power * Delay<sup>2</sup> =</p>
      <p>9.856*10<sup>-6</sup>* (3.13e-10)2</p>
      <p>* 1.63* 10<sup>-12</sup></p>
      <p>= 1.5738994*10<sup>-25</sup></p>
      <table className="dataframe w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
            <th>Memory Cell Area</th>
            <th>Delay</th>
            <th>Power</th>
            <th>FOM</th>
          </tr>
            </thead>
              <tbody>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>9.856*10<sup>-6</sup> </th>
                  <td>3.13e-10s</td>
                  <td>1.63* 10<sup>-12</sup>W</td>
                  <td>1.57*10<sup>-25</sup>             </td>
                </tr>
        </tbody>
        </table>
        <br></br>
        <p>Bitline Capacitance:        </p>
        <p>  Bitline capacitance for a single cell = ˠC0 </p>
        <p>The second ˠC0 is for the transistor used for the bitline conditioning.</p>
        <p>Total bitline capacitance = 16 * ˠC0 = 16 ˠC0.</p>
      

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
      {memory_cell_val}
      {mem_col_val}
      {mem_row_val}
      {full_mem_val}
      {design_metrics}
      {delay}
      {power}
      {fom}
      </section>     
     
    );
}

