"use client";
import Image from "next/image";
import Link from "next/link";

import React, { useRef, useCallback, use, useEffect } from "react";
import { useInView, InView} from "react-intersection-observer";
import { CodeBlock, dracula } from 'react-code-blocks';


  function Code_block1(){
  let text = `import os
from google.colab import drive
import pandas as pd
import requests
from lxml import html
import codecs
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
In [95]:
DRIVE_MOUNT = '/content/gdrive'
drive.mount(DRIVE_MOUNT)

CIS545_FOLDER = os.path.join(DRIVE_MOUNT, 'My Drive', 'CIS545_2020')
path = "/content/gdrive/My Drive/Final Project/demos"
Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}



  function Code_block2(){
  let text = `total_rows = 0;
data_df = pd.DataFrame()
for root, dirs, files in os.walk(path):
  for file in files:
    f=codecs.open(os.path.join(root, file), 'r')
    content = f.read()
    dom_tree = html.fromstring(content)
    path = '/html/body/div[2]/div[1]/div[3]/div[3]/div[1]/div[2]/div[2]/div[2]/div/div[5]/div/table'
    table = dom_tree.xpath(path)[0]
    answer_df = pd.read_html(html.etree.tostring(table,method='html'))[0]
    total_rows = total_rows + len(answer_df.axes[0])
    data_df = data_df.append(answer_df)`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}


  function Code_block3(){
  let text = `  import numpy as np
import math


cleaned_df = data_df.drop(columns=['Date', 'Unnamed: 1', 'Rank', 'Unnamed: 5', 'Rating', 'Unnamed: 21'])
cleaned_df['HS%'] = cleaned_df['HS%'].astype(float).map(lambda x: x / 100)
cleaned_df = cleaned_df.rename(columns={'Map': 'map', 'Score': 'score'})
cleaned_df = cleaned_df.rename(columns={'K': 'kills', 'D': 'deaths'})
cleaned_df = cleaned_df.rename(columns={'A': 'assists', '+/-': 'net_kills'})
cleaned_df = cleaned_df.rename(columns={'HS%': 'hs_percentage', 'ADR': 'damage_per_round'})

cleaned_df['wins'] = cleaned_df['score'].map(lambda x: x[:x.find(':')]).astype(int)
cleaned_df['losses'] = cleaned_df['score'].map(lambda x: x[x.find(':')+1:]).astype(int)
cleaned_df['rounds'] = cleaned_df['wins'] + cleaned_df['losses']
cleaned_df = cleaned_df[(cleaned_df['wins'] >= 15) | (cleaned_df['losses'] >= 15)]

map_labels = dict()

def label_maps(x):
  if not x in map_labels:
    map_labels[x] = len(map_labels)

cleaned_df['map'].apply(label_maps)
cleaned_df['map_label'] = cleaned_df['map'].apply(lambda x: map_labels[x])
cleaned_df = cleaned_df.drop(columns=['map'])

cleaned_df = cleaned_df.drop(columns=['score'])
cleaned_df['net_wins'] = cleaned_df['wins'] - cleaned_df['losses']
cleaned_df['result'] = np.sign(cleaned_df['net_wins']) + 1

cleaned_df['kills_per_round'] = cleaned_df['kills'] / cleaned_df['rounds']
cleaned_df['deaths_per_round'] = cleaned_df['deaths'] / cleaned_df['rounds']
cleaned_df['assists_per_round'] = cleaned_df['assists'] / cleaned_df['rounds']
cleaned_df['net_kills_per_round'] = cleaned_df['net_kills'] / cleaned_df['rounds']
cleaned_df[['1v5', '1v4', '1v3', '1v2', '1v1', '5k', '4k', '3k']] = \
  cleaned_df[['1v5', '1v4', '1v3', '1v2', '1v1', '5k', '4k', '3k']].div(cleaned_df['rounds'], axis=0)

cleaned_df = cleaned_df.drop(columns=['kills', 'deaths', 'assists', 'net_kills'])
cleaned_df['damage_per_round'] = cleaned_df['damage_per_round'] / 100

cleaned_df = cleaned_df.drop(columns=['wins', 'losses', 'net_wins'])

cleaned_df`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}



  function Code_block4(){
  let text = `norm_df = cleaned_df.copy(deep=True)
norm_df = norm_df.drop(columns=['map_label', 'result'])
cleaned_df_mean = norm_df.mean()
cleaned_df_max = norm_df.max()
cleaned_df_min = norm_df.min()
norm_df = (norm_df - norm_df.mean()) / (norm_df.max() - norm_df.min())
norm_df`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block5(){
  let text = `import seaborn as sn
import matplotlib.pyplot as plt

sn.set_style('darkgrid')

plt.figure()
corr_df = norm_df.corrwith(cleaned_df['result'])
corr_plot = sn.barplot(x=corr_df.index, y=corr_df.values)
corr_plot.set_xticklabels(labels=corr_plot.get_xticklabels(), rotation=90);
corr_plot.set_title("Correlation of Features To Win Result")
random_df = cleaned_df.sample(200)
new_labels = ['Loss', 'Tie', 'Win']

plt.figure()
kd_plot_scatter = sn.scatterplot(data=random_df, x='kills_per_round', y='result', hue='result', palette="deep")
kd_plot_reg = sn.regplot(data=random_df, x='kills_per_round', y='result', scatter=False)
kd_plot_reg.set_title("Kills Per Round vs Win Result")
for t, l in zip(kd_plot_scatter.legend_.texts, new_labels): t.set_text(l)


plt.figure()
kd_plot_scatter = sn.scatterplot(data=random_df, x='assists_per_round', y='result', hue='result', palette="deep")
kd_plot_reg = sn.regplot(data=random_df, x='assists_per_round', y='result', scatter=False)
kd_plot_reg.set_title("Assists Per Round vs Win Result")
for t, l in zip(kd_plot_scatter.legend_.texts, new_labels): t.set_text(l)

plt.figure()
kd_plot_scatter = sn.scatterplot(data=random_df, x='deaths_per_round', y='result', hue='result', palette="deep")
kd_plot_reg = sn.regplot(data=random_df, x='deaths_per_round', y='result', scatter=False)
kd_plot_reg.set_title("Deaths Per Round vs Win Result")
for t, l in zip(kd_plot_scatter.legend_.texts, new_labels): t.set_text(l)
plt.figure()
kd_plot_scatter = sn.scatterplot(data=random_df, x='damage_per_round', y='result', hue='result', palette="deep")
kd_plot_reg = sn.regplot(data=random_df, x='damage_per_round', y='result', scatter=False)
kd_plot_reg.set_title("Damage Per Round vs Win Result")
for t, l in zip(kd_plot_scatter.legend_.texts, new_labels): t.set_text(l)

plt.figure()
kd_plot_scatter = sn.scatterplot(data=random_df, x='hs_percentage', y='result', hue='result', palette="deep")
kd_plot_reg = sn.regplot(data=random_df, x='hs_percentage', y='result', scatter=False)
kd_plot_reg.set_title("Headshot Percentage vs Win Result")
for t, l in zip(kd_plot_scatter.legend_.texts, new_labels): t.set_text(l)

plt.figure()
kd_plot_scatter = sn.scatterplot(data=random_df, x='rounds', y='result', hue='result', palette="deep")
kd_plot_reg = sn.regplot(data=random_df, x='rounds', y='result', scatter=False)
kd_plot_reg.set_title("Total Rounds vs Win Result");
for t, l in zip(kd_plot_scatter.legend_.texts, new_labels): t.set_text(l)`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}


  function Code_block6(){
  let text = `from torch.utils.data import TensorDataset, DataLoader
pyTorchData = norm_df.copy(deep=True)
for v in pyTorchData.columns: 
  pyTorchData[v] = pyTorchData[v].astype('float32')


pyTorchData['target'] = cleaned_df['result']
train, validate, test = \
              np.split(pyTorchData.sample(frac=1, random_state=42), 
                       [int(.6*len(pyTorchData)), int(.8*len(pyTorchData))])
              

x_train = train.drop(columns=['target'], axis=1)
x_train = torch.tensor(np.array(x_train).astype('float32'))
y_train = train['target'].values
y_train = torch.tensor(np.array(y_train))
torch.reshape(y_train, (len(x_train), 1))


x_val = validate.drop(columns=['target'], axis=1)
x_val = torch.tensor(np.array(x_val).astype('float32'))
y_val = validate['target'].values
y_val = torch.tensor(np.array(y_val))
torch.reshape(y_val, (len(x_val), 1))


x_test = test.drop(columns=['target'], axis=1)
x_test = torch.tensor(np.array(x_test).astype('float32'))
y_test = test['target'].values
y_test = torch.tensor(np.array(y_test))
torch.reshape(y_test, (len(x_test), 1))


train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
test_data = TensorDataset(x_test, y_test )
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 10, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 10, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 10, shuffle = True)`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}


  function Code_block7(){
  let text = `def create_fnn(in_size, num_classes):

    net = nn.Sequential(
        
        nn.Linear(input_size, 15),
        nn.ReLU(),
        nn.Dropout(.5),
        nn.Linear(15, 8),
        nn.ReLU(),
        nn.Dropout(.5),
        nn.Linear(8, 3),
        nn.Softmax(dim=1)
        
    )

    return net`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block8(){
  let text = `win_results = []
def compute_accuracy(net, dataloader, device="cpu"):
    with torch.no_grad():
        tot_correct = 0
        tot_samples = 0
        for dat, lab in dataloader:
            data, labels = dat.to(device), lab.to(device)
            predicitons = net(data)
            pred = torch.argmax(predicitons, dim=1)
            tot_samples += len(data)
            tot_correct += (pred == labels).sum()
    return (tot_correct) / (tot_samples)`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block9(){
  let text = `def train_nn(net, trainloader, validloader, eval_freq, num_epochs, device="cpu"):
  
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_acc = []
    valid_acc = []
    iter_num = 0

    for epoch in range(num_epochs):
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            y = net(data)
            loss = loss_fn(y, labels)
            loss.backward()
            optimizer.step()
            if iter_num % eval_freq == 0:
                train_acc.append(compute_accuracy(net, trainloader, device))
                valid_acc.append(compute_accuracy(net, validloader, device))
            iter_num += 1
    return train_acc, valid_acc`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block10(){
  let text = `def create_acc_curve(train_acc, valid_acc, eval_freq):
    fig, ax = plt.subplots()
    plt.figure()
    x_axis = np.arange(0, len(train_acc) * eval_freq, eval_freq)
    ax.plot(x_axis, train_acc, 
            label='Train final acc: {:.3f}'.format(train_acc[-1]))
    ax.plot(x_axis, valid_acc, 
            label='Validation final acc: {:.3f}'.format(valid_acc[-1]))
    ax.set_title('Accuracy on CSGO STATS Data Set')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('# iterations')
    ax.legend()
    fig.savefig("acc_curves.png")`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}


  function Code_block11(){
  let text = `%%time
if __name__ == '__main__':
    device = "cuda"
    input_size = 15
    num_classes = 3
    eval_freq = 100
    num_epochs = 1000
    nn_model = create_fnn(input_size, num_classes)
    train_acc, valid_acc = train_nn(nn_model, 
                                   train_loader, 
                                   val_loader, 
                                   eval_freq, 
                                   num_epochs, 
                                   device)
    
CPU times: user 4min 31s, sys: 4.39 s, total: 4min 36s
Wall time: 4min 38s`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block12(){
  let text = `create_acc_curve(train_acc, valid_acc, eval_freq);
`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}


  function Code_block13(){
  let text = `torch.save(nn_model, "model.pkl")
`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block14(){
  let text = `def compute_accuracy_with_confidence(net, dataloader, device="cpu"):
    with torch.no_grad():
        tot_correct = 0
        tot_samples = 0
        for dat, lab in dataloader:
            data, labels = dat.to(device), lab.to(device)
            predictions = net(data)
            pred = torch.argmax(predictions, dim=1)
            for values in pred.tolist():
              if values == 2:
                for i in range(len(predictions.tolist())):
                  win_results.append((data.tolist()[i], predictions.tolist()[i][values]))
            tot_samples += len(data)
            tot_correct += (pred == labels).sum()
    return (tot_correct) / (tot_samples)

win_results = []
test_accuracy = compute_accuracy_with_confidence(nn_model, test_loader, device=device).item()
sets = ["Training", "Validation", "Test"]
accuracies = [train_acc[len(train_acc) - 1].item(), valid_acc[len(train_acc) - 1].item(), test_accuracy]
accuracies_plot = sn.barplot(x=sets, y=accuracies)
accuracies_plot.set_ylim(.7, .76)
accuracies_plot.set_title("Accuracy of Each Datasest");
for p in accuracies_plot.patches:
  accuracies_plot.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
      ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
      textcoords='offset points')
`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block15(){
  let text = `sets = ["Win", "Loss", "Tie"]

percent_loss = sum(cleaned_df['result'] == 0) / len(cleaned_df)
percent_tie = sum(cleaned_df['result'] == 1) / len(cleaned_df)
percent_win = sum(cleaned_df['result'] == 2) / len(cleaned_df)

precent_result = [percent_loss, percent_tie, percent_win]
precent_result_plot = sn.barplot(x=sets, y=precent_result)
precent_result_plot.set_title("Percent of Total Dataset");
for p in precent_result_plot.patches:
  precent_result_plot.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
      ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
      textcoords='offset points')
`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}

  function Code_block16(){
  let text = `win_results.sort(key=lambda x: x[1], reverse=True)
sorted_results = [result[0] for result in win_results]
sorted_result_df = pd.DataFrame(sorted_results)
sorted_result_df.columns = [
  'hs_percentage',
  'damage_per_round',
  '1v5',
  '1v4',
  '1v3',
  '1v2',
  '1v1',
  '5k',
  '4k',
  '3k',
  'rounds',
  'kills_per_round',
  'deaths_per_round',
  'assists_per_round',
  'net_kills_per_round'
]
sorted_result_df = (sorted_result_df * (cleaned_df_max - cleaned_df_min))
sorted_result_df = sorted_result_df + cleaned_df_mean
sorted_result_df = sorted_result_df.round(decimals = 2)
sorted_result_df['win_prob'] = win_results
sorted_result_df['win_prob'] = sorted_result_df['win_prob'].apply(lambda x: x[1])
sorted_result_df
`

  return (<CodeBlock text={text} language="python" theme={dracula}></CodeBlock>)
}



 function Section1(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (     
    <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div ref={ref}> <h1>CSGO Win Predictor</h1>
    <Image src="csgo_predictor/csgo_header.png" alt={""}></Image>
    <p>First and foremost this project ties together our interest in the highly popular Counter-Strike: Global Offensive with data analysis. Though the game is still evolving both at the professional and casual levels, there are not many accessible endeavors to understand the impacts of certain game specific statistics on the overall outcome of each individual match. We want to explore what statistics affect the probability of a game resulting in a win, draw, or loss for that player. Through pulling statstics from csgostats.gg we will develop a model to attempt to accurately predict the outcome of a match given an individuals player statistics.</p>
    
    </div>
    </InView>);

}


 function Section2(){
    
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return ( 
  
    <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div ref={ref}><h1>Set Up</h1>
      
    <p>First we import all our necessary modules and mount our google drive folder where we store our data.</p>
    <Code_block1></Code_block1>
    </div>
    </InView>);

}

 function Section3(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (
    <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div ref={ref}><h1>Extract data from HTML</h1>
    <p>We downloaded the HTML files from players&apos; match history pages on csgostats.gg to our drive folder, so we open them here. Each player&apos;s HTML file stores data for their games in a table, which we read onto a Pandas dataframe. Tables within the HTML files are accessed by using the XPATH that leads to each of these tables, which are consistent within each of the files. We then append these tables together to get our raw data dataframe.</p>
    <Code_block2></Code_block2>
    </div></InView>);

}

 function Section4(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return ( 
    <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div ref={ref}><h1>Clean Data Using Pandas</h1>
    <p>We want to clean up our raw data so that we can visualize correlations and so that our data is usable for training.</p>
    <br></br>
    <p>First we drop the columns we don&apos;t want to use, data, rank, and rating (the score from csgostats.gg own in-house rating system for that player&apos;s performance) and rename the remaining columns.</p>
    <br></br>
    <p>Next, since csgostats.gg stores the round win-loss ratio for each game in the form of &quot;WW:LL&quot;, we split that into two columns, a rounds won column and a rounds lost column. After that, we are able to add a column labeling the game as a win, draw, or loss, depending on the sign of the differences between the wins column and the losses column. We also find the length of the game by adding the number of rounds won and the number of rounds loss, and drop games that were not played to completion, i.e. it was the neither case that one team won 16 rounds nor that both teams won 15 rounds, drawing the game.</p>
    <br></br>
    <p>At first we labeled the maps with our numeric value with the hope that it would be usable for training. However, we later came to the conclusion that we did not want this column, so we drop it.</p>
    <br></br>
    <p>Finally, we divide all of our data columns by the length of the game, convert percentages to floating point representation, add an additional feature, net_kills, which is the difference between kills and deaths, and drop columns we no longer need.</p>
    
    
    <Code_block3></Code_block3>
    <br></br>
    <table className="dataframe w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
  <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
    <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
      <th></th>
      <th>hs_percentage</th>
      <th>damage_per_round</th>
      <th>1v5</th>
      <th>1v4</th>
      <th>1v3</th>
      <th>1v2</th>
      <th>1v1</th>
      <th>5k</th>
      <th>4k</th>
      <th>3k</th>
      <th>rounds</th>
      <th>map_label</th>
      <th>result</th>
      <th>kills_per_round</th>
      <th>deaths_per_round</th>
      <th>assists_per_round</th>
      <th>net_kills_per_round</th>
    </tr>
  </thead>
  <tbody>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>0</th>
      <td>0.22</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0.409091</td>
      <td>0.863636</td>
      <td>0.181818</td>
      <td>-0.454545</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>1</th>
      <td>0.32</td>
      <td>0.97</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.107143</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>0.892857</td>
      <td>0.678571</td>
      <td>0.107143</td>
      <td>0.214286</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>2</th>
      <td>0.50</td>
      <td>0.76</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.066667</td>
      <td>30</td>
      <td>2</td>
      <td>1</td>
      <td>0.733333</td>
      <td>0.733333</td>
      <td>0.033333</td>
      <td>0.000000</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>3</th>
      <td>0.16</td>
      <td>0.83</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>25</td>
      <td>3</td>
      <td>2</td>
      <td>0.760000</td>
      <td>0.600000</td>
      <td>0.280000</td>
      <td>0.160000</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>4</th>
      <td>0.50</td>
      <td>0.77</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.043478</td>
      <td>23</td>
      <td>2</td>
      <td>0</td>
      <td>0.695652</td>
      <td>0.695652</td>
      <td>0.130435</td>
      <td>0.000000</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>47</th>
      <td>0.33</td>
      <td>0.83</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.033333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.033333</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
      <td>0.800000</td>
      <td>0.633333</td>
      <td>0.133333</td>
      <td>0.166667</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>48</th>
      <td>0.11</td>
      <td>0.69</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.055556</td>
      <td>18</td>
      <td>6</td>
      <td>0</td>
      <td>0.500000</td>
      <td>0.944444</td>
      <td>0.055556</td>
      <td>-0.444444</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>49</th>
      <td>0.13</td>
      <td>0.68</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>27</td>
      <td>6</td>
      <td>0</td>
      <td>0.592593</td>
      <td>0.777778</td>
      <td>0.111111</td>
      <td>-0.185185</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>50</th>
      <td>0.17</td>
      <td>0.88</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>28</td>
      <td>4</td>
      <td>0</td>
      <td>0.821429</td>
      <td>0.750000</td>
      <td>0.107143</td>
      <td>0.071429</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>51</th>
      <td>0.19</td>
      <td>0.85</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.058824</td>
      <td>17</td>
      <td>7</td>
      <td>2</td>
      <td>0.941176</td>
      <td>0.352941</td>
      <td>0.235294</td>
      <td>0.588235</td>
    </tr>
  </tbody>
</table>
    </div>
    </InView>);

}

 function Section5(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (      
    <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div ref={ref}><h1>Normalize Data</h1>
    <p>In this section we mean normalize our data.</p>
    <Code_block4></Code_block4>
    <br></br>
    <table className="dataframe w-full text-sm text-left rtl:text-right text-gray-500 dark:text-gray-400">
  <thead>
    <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
      <th></th>
      <th>hs_percentage</th>
      <th>damage_per_round</th>
      <th>1v5</th>
      <th>1v4</th>
      <th>1v3</th>
      <th>1v2</th>
      <th>1v1</th>
      <th>5k</th>
      <th>4k</th>
      <th>3k</th>
      <th>rounds</th>
      <th>kills_per_round</th>
      <th>deaths_per_round</th>
      <th>assists_per_round</th>
      <th>net_kills_per_round</th>
    </tr>
  </thead>
  <tbody>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>0</th>
      <td>-0.197254</td>
      <td>-0.155167</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.002985</td>
      <td>-0.261951</td>
      <td>-0.182525</td>
      <td>0.177742</td>
      <td>0.022185</td>
      <td>-0.218266</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>1</th>
      <td>-0.083617</td>
      <td>0.073094</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.195761</td>
      <td>0.166621</td>
      <td>0.068661</td>
      <td>-0.020541</td>
      <td>-0.104188</td>
      <td>0.063869</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>2</th>
      <td>0.120928</td>
      <td>-0.041036</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.069273</td>
      <td>0.309478</td>
      <td>-0.014169</td>
      <td>0.038132</td>
      <td>-0.229097</td>
      <td>-0.026524</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>3</th>
      <td>-0.265436</td>
      <td>-0.002993</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.139060</td>
      <td>-0.047665</td>
      <td>-0.000323</td>
      <td>-0.104725</td>
      <td>0.188339</td>
      <td>0.040969</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>4</th>
      <td>0.120928</td>
      <td>-0.035601</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>0.285080</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.003191</td>
      <td>-0.190522</td>
      <td>-0.033734</td>
      <td>-0.002241</td>
      <td>-0.064771</td>
      <td>-0.026524</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700"> 
      <th>47</th>
      <td>-0.072254</td>
      <td>-0.002993</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>0.109862</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.034894</td>
      <td>0.309478</td>
      <td>0.020447</td>
      <td>-0.069011</td>
      <td>-0.059866</td>
      <td>0.043781</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>48</th>
      <td>-0.322254</td>
      <td>-0.079080</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.034551</td>
      <td>-0.547665</td>
      <td>-0.135323</td>
      <td>0.264323</td>
      <td>-0.191490</td>
      <td>-0.214005</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>49</th>
      <td>-0.299527</td>
      <td>-0.084514</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.139060</td>
      <td>0.095192</td>
      <td>-0.087246</td>
      <td>0.085751</td>
      <td>-0.097473</td>
      <td>-0.104641</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>50</th>
      <td>-0.254072</td>
      <td>0.024181</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>0.122560</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.139060</td>
      <td>0.166621</td>
      <td>0.031573</td>
      <td>0.055989</td>
      <td>-0.104188</td>
      <td>0.003607</td>
    </tr>
    <tr className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>51</th>
      <td>-0.231345</td>
      <td>0.007877</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.044763</td>
      <td>-0.619093</td>
      <td>0.093750</td>
      <td>-0.369431</td>
      <td>0.112683</td>
      <td>0.221612</td>
    </tr>
  </tbody>
</table>
    </div>
    </InView>);

}

 function Section6(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (  <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}> 
  <div ref={ref}><h1>Initial Visualization</h1>
    <p>Here we plot our columns against the label so we can better visualize the data.</p>
    <br></br>
    <p>As with our intuition, average kills, assists, and damage per round are positively correlated with winning, and dying is negatively correlated with winning.</p>
    <br></br>
    <p>Interestingly, however, our data shows that headshot percentage is negatively correlated with winning. We also found that the length of the game had little correlation with winning, though shorter games were slightly more likely to result in a win.</p>
    <br></br>

    <div className="grid grid-cols-[repeat(auto-fill,minmax(6rem,25rem))] gap-x-16 gap-y-10  justify-center">
      <Image src="csgo_predictor/CorrelationOfFeatuersToWinResult.png" alt={""}></Image>
      <Image src="csgo_predictor/KillsPerRoundVs.png" alt={""}></Image>
      <Image src="csgo_predictor/AssitsPerRoundVs.png" alt={""}></Image>
      <Image src="csgo_predictor/DeathsPerRoundVs.png" alt={""}></Image>
      <Image src="csgo_predictor/DamagePerRoundVs.png" alt={""}></Image>
      <Image src="csgo_predictor/HsPerRoundVs.png" alt={""}></Image>
      <Image src="csgo_predictor/TotalRoundsVs.png" alt={""}></Image>

    </div>
    <br></br>
    
    <Code_block5></Code_block5>
    
    </div></InView>
    );

}

 function Section7(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (  <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
     <div ref={ref}><h1>Transform Data for Model</h1>
    <p>Here we reshape our dataframe for training and testing our model. We also set up our data loaders, with batch size 10. The pandas dataframe needs to be shaped such that the pytorch network is able process the data. Furthermore, the data is split up into train, validation, and test sets. With the train set using 60% of the dataset.</p>
    <Code_block6></Code_block6>
    </div></InView>);

}

 function Section8(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (    <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
      <div ref={ref}><h1>Set Up Feedforward Neural Network Model</h1>
    <p>We tried multiple setups of feedforward neural networks with varying amounts of hidden layers, each activated by a ReLU function and a dropout layer with probability .5. Since we only had 15 input features and 3 output classes, we chose between 32 and 4 nodes as the sizes for our hidden layers. We also set up our training loop and other helper functions here.</p>
    <Code_block7></Code_block7>
    <br></br>
    <Code_block8></Code_block8>
    <br></br>
    <Code_block9></Code_block9>
    <br></br>
    <Code_block10></Code_block10>
    </div>
    </InView>);

}

 function Section9(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (<InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div  ref={ref}>
    <p>We train our models for 1000 epochs.</p>
    <Code_block11></Code_block11>
    <br></br>
    <Code_block12></Code_block12>

    <Image src="csgo_predictor/ModelAccuracy.png" alt={""}></Image>

    </div>
    </InView>);

}

 function Section10(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (<InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
  <div ref={ref}><h1>Analysis and Evaluation</h1>
    <p>Here we verify the soundness of our models by checking the accuracy of our models against the test set. Furthermore, we find the amount our data label skews for our classes to ensure that our models perform better than choosing randomly. Below we are able to see the accuracy of our model on our three data sets. Furthermore, as no one result dominates, the model providing an accuracy around 70% shows that it preforms better than a simple random selection.</p>
    <Code_block13></Code_block13>
    <br></br>
    <Code_block14></Code_block14>
    <div className="grid grid-cols-[repeat(auto-fill,minmax(6rem,25rem))] gap-x-16 gap-y-10  justify-center">
      <Image src="csgo_predictor/AccuracryPerDataset.png" alt={""}></Image>
      <Image src="csgo_predictor/PercentOfTotalDataset.png" alt={""}></Image>   
    </div>
      
    <p>Here we sort our predictions to find the games that our model predicted with the highest probability of being a win. Since our data was normalized, we perform those operations in reverse for this to be more readable.</p>
    <Code_block15></Code_block15>
    <br></br>
    <table>
  <thead>
    <tr className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-700 dark:text-gray-400">
      <th></th>
      <th>hs_percentage</th>
      <th>damage_per_round</th>
      <th>1v5</th>
      <th>1v4</th>
      <th>1v3</th>
      <th>1v2</th>
      <th>1v1</th>
      <th>5k</th>
      <th>4k</th>
      <th>3k</th>
      <th>rounds</th>
      <th>kills_per_round</th>
      <th>deaths_per_round</th>
      <th>assists_per_round</th>
      <th>net_kills_per_round</th>
    </tr>
  </thead>
  <tbody >
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>0</th>
      <td>-0.197254</td>
      <td>-0.155167</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.002985</td>
      <td>-0.261951</td>
      <td>-0.182525</td>
      <td>0.177742</td>
      <td>0.022185</td>
      <td>-0.218266</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>1</th>
      <td>-0.083617</td>
      <td>0.073094</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.195761</td>
      <td>0.166621</td>
      <td>0.068661</td>
      <td>-0.020541</td>
      <td>-0.104188</td>
      <td>0.063869</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>2</th>
      <td>0.120928</td>
      <td>-0.041036</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.069273</td>
      <td>0.309478</td>
      <td>-0.014169</td>
      <td>0.038132</td>
      <td>-0.229097</td>
      <td>-0.026524</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>3</th>
      <td>-0.265436</td>
      <td>-0.002993</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.139060</td>
      <td>-0.047665</td>
      <td>-0.000323</td>
      <td>-0.104725</td>
      <td>0.188339</td>
      <td>0.040969</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>4</th>
      <td>0.120928</td>
      <td>-0.035601</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>0.285080</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.003191</td>
      <td>-0.190522</td>
      <td>-0.033734</td>
      <td>-0.002241</td>
      <td>-0.064771</td>
      <td>-0.026524</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>47</th>
      <td>-0.072254</td>
      <td>-0.002993</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>0.109862</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.034894</td>
      <td>0.309478</td>
      <td>0.020447</td>
      <td>-0.069011</td>
      <td>-0.059866</td>
      <td>0.043781</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>48</th>
      <td>-0.322254</td>
      <td>-0.079080</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.034551</td>
      <td>-0.547665</td>
      <td>-0.135323</td>
      <td>0.264323</td>
      <td>-0.191490</td>
      <td>-0.214005</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>49</th>
      <td>-0.299527</td>
      <td>-0.084514</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.139060</td>
      <td>0.095192</td>
      <td>-0.087246</td>
      <td>0.085751</td>
      <td>-0.097473</td>
      <td>-0.104641</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>50</th>
      <td>-0.254072</td>
      <td>0.024181</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>0.122560</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>-0.139060</td>
      <td>0.166621</td>
      <td>0.031573</td>
      <td>0.055989</td>
      <td>-0.104188</td>
      <td>0.003607</td>
    </tr>
    <tr  className="bg-white border-b dark:bg-gray-800 dark:border-gray-700">
      <th>51</th>
      <td>-0.231345</td>
      <td>0.007877</td>
      <td>-0.001969</td>
      <td>-0.010322</td>
      <td>-0.01927</td>
      <td>-0.048254</td>
      <td>-0.067916</td>
      <td>-0.011538</td>
      <td>-0.037175</td>
      <td>0.044763</td>
      <td>-0.619093</td>
      <td>0.093750</td>
      <td>-0.369431</td>
      <td>0.112683</td>
      <td>0.221612</td>
    </tr>
  </tbody>
</table>
    <p>The statistics of these games and the probability that they are labeled a victory is in line with our intuition that the player&apos;s statline is a good indicator for the result of the game. For example, in the first game, the player had 24 kills, 7 assists, and 16 deaths, a good performance. However, consider the third game from the top; this player did not perform particularly well, but the game is still labeled as a highly probably win, indicating that there is more to winning than just the player&apos;s statline.</p>
    <Code_block16></Code_block16>
    </div></InView>);

}


 function Section11(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (   <InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>

  <div ref={ref}><h2>Model with Hidden Layers of Size 32, 32, 32, 32, 16, 16, 8</h2>
    <Image src="csgo_predictor/Model1.png" alt={""}></Image>   
    <h2>Model with Hidden Layers of Size 15, 8</h2>
    <Image src="csgo_predictor/Model2.png" alt={""}></Image>   
    <h2>Model with No Hidden Layers (Linear Regression)</h2>
    <Image src="csgo_predictor/Model3.png" alt={""}></Image>   
    </div></InView>);

}


 function Section12(){
  const { ref, inView, entry } = useInView(
    {
      /* Optional config */
      threshold: 0.1,  // Trigger when 10% of the element is in view
      triggerOnce: true,  // Trigger only once
    }
  );

  return (<InView as="div" onChange={(inView, entry) => {entry.target.children[0].classList.add('show')}} className={`transition-opacity duration-1000 ${inView ? 'opacity-100 shown' : 'opacity-0'}`}>
    <div ref={ref}>
    <h1>Conclusions</h1>
    
    <p>At this point we concluded that linear regression would provide a better means of predicting the outcome of the game than a neural net would with the data collected. We believe this is because we do not have enough features or sufficiently complex characteristics in our data. Being limited by the data we can collect from sites like csgostats.gg, we do not believe we could get much better predictions from our current level of information. In particular, as CS:GO is a team-based 5v5 game, simply collecting data from a single player&apos;s impact on the game does not completely represent the quality of their team&apos;s play. Furthermore, that player&apos;s explicit statistics may not capture their impact on their teammates, such as communication, leadership, or other forms of assistance.</p>
    </div>
    </InView>);

}









export default function Csgo_Project() {



  return (
      <section className="">
      <Section1/>
      <Section2/>
      <Section3/>
      <Section4/>
      <Section5/>
      <Section6/>
      <Section7/>
      <Section8/>
      <Section9/>
      <Section10/>
      <Section11/>
      <Section12/>
      

      </section>     
     
    );
}

