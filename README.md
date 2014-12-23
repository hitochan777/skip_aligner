aligner
==========

`aligner` is a word aligner which has inherited some of the basic functions that fast_align has. 

If you use this software, please cite:
* [Chris Dyer](http://www.cs.cmu.edu/~cdyer), [Victor Chahuneau](http://victor.chahuneau.fr), and [Noah A. Smith](http://www.cs.cmu.edu/~nasmith). (2013). [A Simple, Fast, and Effective Reparameterization of IBM Model 2](http://www.ark.cs.cmu.edu/cdyer/fast_valign.pdf). In *Proc. of NAACL*.

The source code in this repository is provided under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

## Input format

There are two input formats. The first format is identical to the original aligner(i.e. fast_algin)
### Putting parallel sentences to one file.
Input to `aligner` must be tokenized and aligned into parallel sentences. Each line is a source language sentence and its target language translation, separated by a triple pipe symbol with leading and trailing white space (` ||| `). An example 3-sentence German–English parallel corpus is:

    doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
    neue Modelle werden erprobt . ||| new models are being tested .
    doch fehlen uns neue Ressourcen . ||| but we lack new resources .
### Using different files for source and target language corpuses.
If you use this format, make sure to specify that you use two input files, using `--finput` and `--einupt` options; otherwise the program will not work.

## Compiling and using `aligner`

Building `aligner` requires only a C++ compiler; this can be done by typing `make` at the command line prompt. Run `aligner` to see a list of command line options.

`aligner` generates *asymmetric* alignments (i.e., by treating either the left or right language in the parallel corpus as primary language being modeled, slightly different alignments will be generated). The usually recommended way to generate *source–target* (left language–right language) alignments is:

    ./aligner -i text.fr-en -d -o -v > forward.align

The usually recommended way to generate *target–source* alignments is to just add the `-r` (“reverse”) option:

    ./aligner -i text.fr-en -d -o -v -r > reverse.align

Using [other](http://www.cdec-decoder.org/) [tools](http://www.statmt.org/moses/), the generated forward and reverse alignments can be *symmetrized* into a (often higher quality) single alignment using intersection or union operations, as well as using a variety of more specialized heuristic criteria.

## Output

`aligner` produces outputs in the widely-used `i-j` “Pharaoh format,” where a pair `i-j` indicates that the <i>i</i>th word (zero-indexed) of the left language (by convention, the *source* language) is aligned to the <i>j</i>th word of the right sentence (by convention, the *target* language). For example, a good alignment of the above German–English corpus would be:

    0-0 1-1 2-4 3-2 4-3 5-5 6-6
    0-0 1-1 2-2 2-3 3-4 4-5
    0-0 1-2 2-1 3-3 4-4 5-5

## Acknowledgements

The development of this software was sponsored in part by the U.S. Army Research Laboratory and the U.S. Army Research Ofﬁce under contract/grant number W911NF-10-1-0533.

