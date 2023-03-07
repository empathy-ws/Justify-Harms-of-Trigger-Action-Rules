# Justify Harms of Trigger-Action Rules
This repository contains the supplementary material for the paper "Hybrid Prompt Learning for Generating Justifications of Security Risks in Automation Rules" submitted to ACM Transactions on Intelligent Systems and Technology. 

This material comprises the dataset employed to train the language models and the source codes useful for the repeatability of the experiments.

# Creators

Bernardo Breve (bbreve@unisa.it), Gaetano Cimino (gcimino@unisa.it), Vincenzo Deufemia (deufemia@unisa.it)

University of Salerno

# Dataset Information
The dataset on which we performed our study was initially collected in 2017 by Mi <i> et al. </i> through a scraping process [1] (<a href="https://www-users.cse.umn.edu/~fengqian/ifttt_measurement/">download</a>). The dataset comprised over 320,000 rules (also known as applets) gathered over a six-month period, with each applet consisting of: $(i)$ a title (<i>Title</i>), $(ii)$ a description of its behavior (<i>Desc</i>), $(iii)$ the trigger component divided into its trigger (<i>TriggerTitle</i>) and relative channel (<i>TriggerChannelTitle</i>), and $(iv)$ the action component divided into its action (<i>ActionTitle</i>) and relative channel (<i>ActionChannelTitle</i>). To train and evaluate our language models, we considered a subset of 525 applets randomly selected that were classified as harmful by our harm identification algorithm proposed in [2]. In particular, the algorithm can classify an applet according to three classes of harm, namely <i>Personal harm</i>, identifying the applets that may result in the loss or compromise of sensitive data, which is solely due to the user's behavior, <i>Physical harm</i>, which refers to all the applets that may cause physical harm or damage to goods, and the harm is external, i.e., inflicted by third parties, and <i>Cybersecurity harm</i>, encompasses applets that may disrupt online services or distribute malware, and the harm is external as well. The subset of applets comprised 183 applets for the Personal harm class, 153 applets for the Physical harm class, and 189 applets for the Cybersecurity harm class.

<table align="center">
    <tr>
     <td align="center"><b>Attribute Characteristics:</td>
        <td align="center"><i>Categorical</td>
        <td align="center"><b>Number of samples:</td>
        <td align="center"><i>525</td>
    </tr>
    <tr>
        <td align="center"><b>Collection Method:</td>
        <td align="center"><i>Crowd + Authors</td>
        <td align="center"><b>Number of attributes:</td>
        <td align="center"><i>8</td>
    </tr>
    <tr>
        <td align="center"><b>Associated Tasks:</td>
        <td align="center"><i>Natural Language Generation</td>
        <td align="center"><b>Total Annotators:</td>
        <td align="center"><i>13</td>
    </tr>
</table>

We split the dataset into 3 subsets:

<table align="center">
    <tr>
        <td align="center"><b>Split</td>
        <td align="center"><b>Number of samples</td>
        <td align="center"><b>Filename</td>
        <td align="center"><b>Description</td>
    </tr>
     <tr>
        <td align="center">train</td>
        <td align="center">435</td>
        <td align="center"><a href = "https://github.com/empathy-ws/Justify-Harms-of-Trigger-Action-Rules/blob/main/training_dataset.csv"> training_dataset.csv </a></td>
        <td align="center">Model Training</td>
    </tr>
    <tr>
        <td align="center">val</td>
        <td align="center">30</td>
        <td align="center"><a href = "https://github.com/empathy-ws/Justify-Harms-of-Trigger-Action-Rules/blob/main/val_dataset.csv"> validation_dataset.csv </a></td>
        <td align="center">Hyperparameter Tuning</td>
    </tr>
    <tr>
        <td align="center">test</td>
        <td align="center">60</td>
        <td align="center"><a href = "https://github.com/empathy-ws/Justify-Harms-of-Trigger-Action-Rules/blob/main/test_dataset.csv"> test_dataset.csv </a></td>
        <td align="center">Model Testing</td>
    </tr>
</table>

# Attribute Information

<table align="center">
    <tr>
        <td align="center"><b>TriggerTitle:</td>
        <td align="center">is a string representing the name of the trigger that activates the applet</td>
    </tr>
    <tr>
        <td align="center"><b>TriggerChannelTitle:</td>
        <td align="center">is a string representing the name of the trigger channel chosen by the user as a trigger for the applet</td>
    </tr>
    <tr>
        <td align="center"><b>ActionTitle:</td>
        <td align="center">is a string representing the name of the action to execute in response to the trigger</td>
    </tr>
    <tr>
        <td align="center"><b>ActionChannelTitle:</td>
        <td align="center">is a string representing the name of the action channel chosen as an action by the user</td>
    </tr>
    <tr>
        <td align="center"><b>Title:</td>
        <td align="center">is a string containing the title of the applet</td>
    </tr>
    <tr>
        <td align="center"><b>Desc:</td>
        <td align="center">is a string containing a description of the applet's behavior</td>
    </tr>
    <tr>
        <td align="center"><b>Target:</td>
        <td align="center">is a number representing the harm class</td>
    </tr>
    <tr>
        <td align="center"><b>Justification:</td>
        <td align="center">is a natural language sentence that explain the reason why the applet might cause specific harm</td>
    </tr>
</table> 

An annotated sample is shown below.

```
{
    "title": "Blink lights when mentioned on Twitter",
    "description": "Blink lights when you receive a mention on Twitter.",
    "actionChannelTitle": "LIFX",
    "actionTitle": "Blink lights",
    "triggerChannelTitle": "Twitter",
    "triggerTitle": "New mention of you",
    "target": "2",
    "justification": "This rule might cause a Physical harm because an attacker could mention the user on social networks causing the continuously blinking of lights, damaging them."
}
```

# Usage

## Task Definition
In the considered natural language generation task, we assume a user has created an automation rule $R$ using a TAP, and an algorithm has identified a potential harm $H$ related to the execution of $R$. The goal is to generate a sequence of words $J=(j_1,j_2,\ldots,j_N)$ that justifies why $H$ can be caused by $R$. We propose a hybrid prompt learning-based strategy for generating justifications of security risks associated to automation rules. It applies a small change to $R$ by replacing the pre-trained language model embeddings of channel names involved in the rule with embeddings generated by the Channel2Vec strategy. The task performance is evaluated using BLEURT, METEOR and Coleman-Liau index.

## How to compute Channel2Vec representations?
We extract the list of IFTTT channels to be mapped into embeddings from the dataset we proposed in [2] (<a href = "https://github.com/empathy-ws/Harmful-ECA-rules-classifiers">dataset</a>). It contains a variety of categories, ranging from online services and content providers to smart home devices, for a total of 435 channels. Channel descriptions are crawled on the Google Play Store app by running the following command
```
python crawler.py
```
Sentence embeddings of descriptions are calculated by SentenceBERT, whereas we use the cosine similarity to compare the embeddings. The labeled pairs (target channel, context channel) are constructed by running the following command
```
python create_dataset.py
```
Finally, Channel2Vec representations are computed by running the following command
```
python channel2Vec.py
```

## Run Evaluation
We consider two different hard prompts where the input is placed entirely before the slot to be filled. The former, named <i>span-infilling prompt</i>, is a straightforward sequence of natural language tokens following encoded rule information. 
```
[R]. This rule might cause a [H] harm because [J]
```

The latter is a <i> question-answering prompt </i> where the encoded rule information is placed in the middle of the prompt.
```
Why might rule [R] cause a [H] harm? [J]
```

We insert the Channel2Vec representations within the hard prompts, yielding <i> hybrid prompts </i> that allow us to grasp both the rule's context of execution and the rule behavior and generate a technical justification consistent with the harm involved. In particular, we apply hard-soft prompts by augmenting applets with the span-infilling or question-answering prompt, considering the features <i> Desc, Title, TriggerTitle, ActionTitle </i>, and <i> Harm </i> as discrete tokens and treating the <i> TriggerChannelTitle </i> and <i> ActionChannelTitle </i> features (i.e., the channel used for the trigger and the action, respectively) as continuous tokens.

Below are commands for performing GPT-2 model training using the hybrid prompt learning-based strategy
```
python hybrid_qa.py
------------------------------------
python hybrid_span.py
```

# Relevant Papers
[1] Mi, Xianghang, et al. "An empirical characterization of IFTTT: ecosystem, usage, and performance." Proceedings of the 2017 Internet Measurement Conference (2017).

[2] Breve Bernardo, Gaetano Cimino, and Vincenzo Deufemia. "Identifying Security and Privacy Violation Rules in Trigger-Action IoT Platforms with NLP Models." IEEE Internet of Things Journal (2022).
