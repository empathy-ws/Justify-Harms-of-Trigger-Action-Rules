# Justify Harms of Trigger-Action Rules
This repository contains the supplementary material for the paper "Hybrid Prompt Learning for Generating Justifications of Security Risks Raised by Automation Scenarios" submitted to ACM Transactions on Intelligent Systems and Technology Journal.

This material comprises the dataset employed to train the language models and the source codes useful for the repeatability of the experiments.

# Creators

Bernardo Breve (bbreve@unisa.it), Gaetano Cimino (gcimino@unisa.it), Vincenzo Deufemia (deufemia@unisa.it)

University of Salerno

# Dataset Information

<table align="center">
    <tr>
     <td align="center"><b>Dataset Characteristics:</td>
        <td align="center"><i>Textual</td>
        <td align="center"><b>Number of instances:</td>
        <td align="center"><i>525</td>
    </tr>
    <tr>
        <td align="center"><b>Attribute Characteristics:</td>
        <td align="center"><i>Categorical</td>
        <td align="center"><b>Number of attributes:</td>
        <td align="center"><i>8</td>
    </tr>
    <tr>
        <td align="center"><b>Associated Tasks:</td>
        <td align="center"><i>Natural Language Generation</td>
        <td align="center"><b>Number of classes:</td>
        <td align="center"><i>4</td>
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
        <td align="center">is a natural language sentence that explain the reason why an applet might cause specific harm</td>
    </tr>
</table> 
