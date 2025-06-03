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

Time test on 5 Docs:
=== Time spent per segment ===
04-model-forward  :    188.801 s
02-dataset-load   :      3.183 s
01-model-load     :      3.106 s
05C-scatter       :      1.745 s
05B-build-windows :      0.636 s
05E-save          :      0.204 s
05A-snake-fill    :      0.028 s
00-setup          :      0.004 s
05D-slide         :      0.001 s
03-snake-init     :      0.000 s
Unnacounted time  :     about 4s


Time test on 50 Docs:
=== Time spent per segment ===
04-model-forward  :   2924.686 s
05C-scatter       :     43.977 s
05B-build-windows :     11.268 s
05E-save          :      6.298 s
02-dataset-load   :      2.259 s
01-model-load     :      2.062 s
05A-snake-fill    :      0.281 s
05D-slide         :      0.025 s
00-setup          :      0.002 s
03-snake-init     :      0.000 s


Time test without upcasting on 5 docs:
=== Time spent per segment ===
04-model-forward  :    181.235 s
02-dataset-load   :      2.314 s
01-model-load     :      2.272 s
05C-scatter       :      1.719 s
05B-build-windows :      0.636 s
05E-save          :      0.192 s
05A-snake-fill    :      0.014 s
00-setup          :      0.002 s
05D-slide         :      0.001 s
03-snake-init     :      0.000 s


Difference to version using upcasting (arbitrary doc):
Mean   A : 2.593810
Mean   B : 2.593809
Mean   difference (A-B) : 0.000000
Mean |difference|       : 0.000000
Variance of difference  : 5.060902e-10
Max |difference|        : 0.007812  @ index (np.int64(4), np.int64(875))