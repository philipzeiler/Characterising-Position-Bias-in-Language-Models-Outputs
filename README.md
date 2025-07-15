# Code

This is the initial Github repo to document the code for my Bachelors thesis: "Characterising Position Bias in Language Model’s Outputs".

initial testing: documents the first tests I ran to see how I can get the log-likelihood of the output of the Pythia models.

home validation set testing: I will try to adapt the initial code according to the specifications set initially and run a few starting tests on the validation set on my own computer.

sequence_0_NLL_test: I attempt to run the complete evaluation of one single document (doc-0). I explored two different padding strategies: one where I used a left buffer full of random sequences for every single inference, and the other where I generated the padding randomly once, and then only shifted it into the window for every inference. The first option created matrix heat map images which were very noise, but the second seems to show a trend of increasing NLL for positions later in the window.

doc_snake_test: Likely final file. Feeds all documents through the context one after another like a giant snake, whithout any filler. Records these calculated NLL numbers in one .h5 matrix per doc. I tested different functionality and performance improvements extensively and this is where I ended up.

h5_equality_checker: Debug code which checks two .h5 files for equality and if they are not equal outputs some info about the difference.

matrix_checker: Debug code which analyzes a single .h5 matrix and gives some info on it. Used by me early in testing to make sure the matrices where being generated correctly.

histogram: Code I generated in order to generate a histogram of the Pile val set and give some info about the distribution of file sizes.

val_set_explorer: Similar to histogram but gives some more info.

nll_for_all: Generates a graph of the NLLs of all docs over every position within the context. Takes a long time to run for many files but gives good overview of larger trends.

nll_for_no_filler: Similar to nll_for_all but focuses on those tokens which never see any filler (meaning the only context they ever have is from their own doc, never from another).

nll_for_small_doc: Similar to nll_for_all but focuses on those documents which are shorter thann 500 tokens in order to see how how the filler from other random docs effects the NLL.