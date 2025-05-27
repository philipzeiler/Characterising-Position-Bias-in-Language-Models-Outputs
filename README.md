# Code

This is the initial Github repo to document the code for my Bachelors thesis: "Characterising Position Bias in Language Model’s Outputs".

initial testing: documents the first tests I ran to see how I can get the log-likelihood of the output of the Pythia models.

home validation set testing: I will try to adapt the initial code according to the specifications set initially and run a few starting tests on the validation set on my own computer.

sequence_0_NLL_test: I attempt to run the complete evaluation of one single document (doc-0). I explored two different padding strategies: one where I used a left buffer full of random sequences for every single inference, and the other where I generated the padding randomly once, and then only shifted it into the window for every inference. The first option created matrix heat map images which were very noise, but the second seems to show a trend of increasing NLL for positions later in the window.

val_set_explorer: short code created to analyze the validation set. Returned the following:
Documents analysed           : 214,670
Min tokens per doc           : 0
Max tokens per doc           : 981,471
Mean tokens per doc          : 1,587.24
Median tokens per doc        : 438.0
Docs > 2048 tokens           : 27,802
Total tokens in split        : 340,733,398
Tokens in docs < 2048 tokens : 96,282,557 (28.26 % of total)