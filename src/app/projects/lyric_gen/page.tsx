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



export default function LyricGenProject() {
  let title = <>
  <h1>Automatic Lyric Generation with Machine Learning</h1>
  <h2>Abstract:</h2>
  <p>
  &emsp; In this project we apply various machine learning models to the task of generating lyrics given some starting phrase(s). In training the model with a variety of existing song data, we hope to provide a complement to songwriters in order to speed up or provide general inspiration for song production. Our primary contribution involves extracting data from Spotify’s top 50 artists’ songs by cross searching these titles and artists on the Genius lyric database. This trains the model on specifically more relevant and popular song lyrics as opposed to previous works in the area that scrape large quantities of data from Genius without filtering. We are also contributing through the application of several different models(character level, word level LSTMs, and GRUs each with and without audio features) to the input data to compare the strengths and weaknesses of each model. As of now, the LSTM model without audio features achieves the greatest lyric similarity to the training data in terms of lyric structure and form. 

  </p>
  </>

  let introduction = FadeInFactory(<>
  <h2>Introduction</h2>
  <p>&emsp;Automated lyric generation for songwriting has only recently become a more 
    researched field due to the perceived difficulty in emulating human creativity. With the 
    use of more recent RNN models such as the LSTM or GRU however, this task has now 
    become more feasible. In this project, we attempt to expand upon existing lyric 
    generation models through new training data as well as a variety of different recurrent 
    models in an attempt to best replicate the style of human lyrics. The issue at hand could 
    serve as inspiration for artists in writing their own lyrics and act to speed up the overall 
    music production process, helping to push along the music industry for public 
    consumption. </p>
    <p>&emsp;We tackle the issue of generating additional lyrics with an initial 
    sentence/sequence of tokens as the start. We set this input as the window, then 
    iteratively generate additional lyrics we add to this window for input in future iterations. 
    After a set number of iterations(the number of lyrics desired to be generated), this 
    dynamic window serves as the output in the form of a series of tokens. </p>
    <p>&emsp;In addition to the base input, we also train alternative versions of each model(the 
    character level, word level, and GRU), which take an additional audio features vector as 
    input that contributes to the final token generated. For these models, we define the input 
    as the initial sequence of tokens and a desired audio features vector(danceability, 
    liveliness, etc...) to help influence the final token generation.
    </p>
  </>);
  let background = FadeInFactory(<>
    <h2>Background</h2>
    <p>&emsp;Previous works in automated lyric generation lie mainly in the use of RNNs with 
      recently some applications of LSTMs in the area. As noted in Deep Learning in Musical 
      Lyric Generation: An LSTM-Based Approach listed below, some other projects focused 
      mainly on specific genres and structure, such as Potash et. al (2015) in creating a rap 
      lyric generator with LSTMs focused solely on a specific artist and rhyming. Others such 
      as Watanabe et. al (2018) have made developments in lyric generation with melody 
      input, albeit using more traditional RNNs. The work done by Gill, Lee, and Marwell 
      (2020) expands upon these by training models on a variety of genres through 
      differentiating the training data, achieving more specialized and personalized lyric 
      generation. 
      </p>
      <p>&emsp;We aim to further build upon these works in an attempt to increase the quality of 
      generated lyrics by introducing various new features into the training data as provided 
      through the Spotify API. These include the danceability, liveliness, or energy of the song 
      lyrics as defined by Spotify. Furthermore, we intend to attempt and use GRUs as a 
      different approach to such lyric generation problems to see whether performance can 
      be improved. 
      </p>
      <p>Prior Works: </p>
      <section>
        <ul>
          <li>Deep Learning in Musical Lyric Generation: An LSTM-Based Approach by Gill, 
          Lee, and Marwell
          <section>
            <ul>
              <li>
              Code repository: <br />
              https://github.com/danielhorizon/lyrics-genreation. 
              </li>
              <li>
              The paper itself is also available, as a PDF: <br/>
              <Link href="https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1000&context=yurj">https://elischolar.library.yale.edu/cgi/viewcontent.cgi?article=1000&context=yurj</Link>
              </li>
            </ul>
            We are closely building off of some of the model setup in this paper that similarly 
            applies LSTMs to general lyric generation. The piece looks mainly at scraped 
            data from Genius without any filters, and thus translates relatively well to our 
            project. 
            </section> 
          </li>
          
        </ul>
      </section>
    
  </>)

  let  summary = FadeInFactory(<>
    <h2>Summary of Our Contributions </h2>
    <p>Our algorithmic contribution consists of the use of new and different model 
    architectures towards the problem of lyric generation. This involves LSTM and GRU 
    models with additional parameters to the embedded layer. </p>
    <p>Our data contribution consists of the use of a new dataset derived from the songs 
    of the top 50 artists on Spotify, of which we cross search the actual lyrics on Genius. 
    This dataset differs in song relevance and popularity from previous works.
    </p>
    
  </>)

  let description_of_contrib_1 = FadeInFactory(<>
  <h2>Detailed Description of Contributions </h2>
  <section>
    <h3>Algorithm Contribution 
    </h3>
    <p>&emsp;Our contributions are models that expand on the long short-term memory (LSTM) 
      model for generating lyrics. As a baseline, we created two LSTM models, one that 
      generates one character at a time based on a context of characters, and one that is 
      word level, and one that generates one word at a time based on a context of entire 
      words. These are the character-level LSTM model and the word-level LSTM model, 
      respectively.</p>
    <p>&emsp;Besides the word-level LSTM model, we also created a gated recurrent unit 
      (GRU) model with the same input and output. This is because GRU models provide 
      similar performance for the task of natural language processing while having decreased 
      training time. This is because GRU models are simpler than LSTM models since the 
      former has only two gates (reset and update), while the latter has three gates (input, 
    output, and forget).</p>
    <p>&emsp;Expanding upon the word-level LSTM model and the word-level GRU model, we 
      created two more models by incorporating the audio features of the songs used in the 
      training data. This is done by concatenating the audio features of the song to the input 
      of the fully connected layers. Audio features are attributes of a given song provided by 
      Spotify, such as danceability, valence, energy, or tempo. </p>
      <img src="lyric_gen/arch.png"></img>
  </section>
  </>)
  
  let description_of_contrib_2 = FadeInFactory(<>
    <h3>Data Contribution </h3>
    <p>Our data collection method was completely different than what was outlined in 
    the paper. When collecting data we used the spotify API to collect the artists of the 50 
    most popular songs in the United States. After collecting the artists, we get the top 
    songs of the artists and their audio features from the Spotify API. After getting these 
    songs we get the lyrics using the Genius API. We then take all that information and add 
    them to the same data frame that will be used for training after being cleaned. </p>
        <img src="lyric_gen/data.png"></img>
        <br></br>
        <p>After creating the data frame of all data, we check the language of each song by 
    using the langdetect module. We exclude all songs that are not in english as our current 
    goal is to only use songs that are in English. We also remove characters that are not 
    alphabetical, spaces, or punctuation marks. </p>
        <p>We also tried a different method for preprocessing the training data. In the paper 
    that we use as our primary reference, all the songs in the training set are concatenated 
    together into a single string, from which 10-grams are generated. However, this would 
    mean that some words are generated from a context with tokens from multiple songs. 
    To account for this, we prune the contexts that cross between songs, which ensures that 
    generated words only use contexts with exclusively words from a single song. </p>
    </>)
  let methods = FadeInFactory(<>
  <h2>Methods</h2>
    <p>Our proposed approach based our models on models from “Deep Learning in 
  Musical Lyric Generation: An LSTM-Based Approach” [Gill, 2020]. This includes a 
  character level and world level LSTM model for text generation. We then expanded this 
  domain into a GRU. Our goal was to determine which of these approaches had the best 
  performance for training efficiency for our purposes. While training we consider word 
  accuracy with cross entropy loss. After training, we used numerical approaches as well 
  as holistics methods to determine the best model; we also calculated the intersection
  over union of the set of trigrams that appear in the training data and the set of trigrams 
  that appear in the generated data. By analyzing these metrics after training we are able 
  to understand more clearly the performance of the model. 
  </p>
  <p>For 2 of the models, we will add the audio features into the linear layer right 
  before our output to allow for the audio features to affect the final words of our lyrics 
  </p>
  <p>The models were trained on slices of lyrics from each individual song. For the 
  character level model, for the training data a sequence of 50 characters were input into 
  the LSTM and the output was the next predicted character. For the word level models, 
  the sequence length was 10 words and the output was the next predicted word. 
  </p>
  <p>The training data differs from the yale paper as this training set does not 
  concatenate all lyrics together, but rather looks at slices from each song, stopping when 
  there are no more characters left. The prediction is based on the probability distribution 
  created from the full connected layers which has the same number of outputs as we 
  have total unique words in all songs combined. The predicted word is the node with the 
  greatest probability after softmax is applied to the output layer. </p>
  <p>The loss function used during optimization was cross entropy loss because our 
  problem became a multi-class classification problem. The learning rate used for all 
  models was .002, we had a large set of training, so it would be inappropriate to use a 
  large learning rate as we would not be able to converge to minima. We limited our total 
  number of epochs to 20 as we did not want to spend too much time on training, but 
  rather wanted to spend our time developing multiple models.</p>
  </>)  
  
  let experiments = FadeInFactory(<>
    <h2>Experiments and Results </h2>
    <p>To evaluate the performance of our models, we collect two sets of metrics for 
each model. The first is a vector of holistic numerical metrics, and the second is an 
n-gram based approach. </p>
    <p>To find the holistic metrics for any given song, we calculate the average line 
length in words, number of unique words, how frequently a sentence begins with a first 
person noun minus how frequently a sentence begins with a second person noun,and 
how many times a word repeats. The latter three metrics are normalized. We calculate 
this 4-vector y<sub>training</sub> for each song in our training data. We also calculate a 4-vector
y <sub>generated</sub> for each of the outputs of the models. Finally, we find the cosine similarity by calculating s = y<sub>training</sub>  * y<sub>generated</sub> / (||y<sub>training</sub>|| *|| y<sub>generated</sub>||)
 </p>
    <p>We also find the tri-grams of each song in the training data. After that, we find the 
tri-grams of each of the outputs of the models, and then calculate the intersection over 
union of the sets of trigrams, or the number of tri-grams that appear in both the
generated song and a training song divided by the number of tri-grams that appear in 
either. For this, we also find both maximum and average loU for each model with the 
entire training data. </p>
    <p>Given the input sentence of &quot;i remember dad walking out that door\nbut i &quot;, we calculated these metrics for each model.  Our results are shown in the table below: 
    </p>
    
    <table className="dataframe w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
          <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
            <th>Model</th>
            <th>Mean Cos. Similarity</th>
            <th>Mean Trigram loU </th>
            <th>Max Cos. Similarity</th>
            <th>Max Trigram loU </th>
          </tr>
            </thead>
              <tbody>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>LSTM (Char) </th> <th>0.9399888</th>  <th>0.002027704 </th> <th>0.9996270 </th> <th>0.008152174 </th>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>GRU (Word) </th><th> 0.9437032 </th><th>0.003837726 </th><th> 1.000000 </th><th> 0.0184621 </th>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>LSTM (Word) </th><th>0.9435447</th><th> 0.006104015</th><th> 1.000000</th><th> 0.04008439 
                  </th>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>GRU (Features)</th><th>  0.8441254</th><th> 0.002911991</th><th> 0.998306</th><th> 0.01251739 </th>
                </tr>
                <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
                  <th>LSTM (Features) </th><th> 0.9575105</th><th> 0.003692427</th><th> 0.999822</th><th> 0.02988792 </th>
                </tr>
        </tbody>
        </table>

    <p>The cosine similarity of the holistic metrics ensure that the generated lyrics are 
reasonable, e.g. they have similar word repetitiveness or line length as the songs in the 
data set. The trigram loU measures how well the models arrange words into strings that 
resemble lyrics from the training set. From these results, the LSTM model appears to 
have performed the best. </p>
    </>)  
    
  let compute = FadeInFactory(<>
    <h2>Compute/Other Resources Used </h2>
    <p>All computation was done within the Google Colab notebook environment using 
    their Google compute engine backend GPUs. Training epochs and quantity were limited 
    as to not require additional AWS resources. </p>
    </>)  

  let conclusion = FadeInFactory(<>
    <h2>Conclusions </h2>
    <p>Through this project, we compared and contrasted the lyric generation abilities of 
    several different model architectures utilizing a new training dataset. As described in the 
    results section above, the word level LSTM without audio features seemed to perform 
    the best on average with respect to our designed metrics. In the technical community, 
    we hope this result can serve to help others looking into expanding upon this problem to 
    choose better performing models. In general, we hope these trained models can serve 
    as a tool for artists to better develop and communicate themselves through song lyrics. 
    However, we must note that overreliance on such tools may actually be detrimental to 
    creative tendencies, and hence warn against it. </p>
    </>)  
  return (
    <>
    {title}
      <section>
      <br/>
        {introduction}
        <br/>
        {background}
        <br/>
        {summary}
        <br/>
        {description_of_contrib_1}
        <br/>
        {description_of_contrib_2}
        <br/>
        {experiments}
        <br/>
        {compute}
        <br/>
        {conclusion}
      </section>     
    </>
    );
}

