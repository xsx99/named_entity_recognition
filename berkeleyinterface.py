# BerkeleyInterface.berkeleyinterface
# Main functionality of the interface
# author: XuL


"""
Python interface to the Berkeley Parser

This has the advantage over other implementations which essentially automate a
call to the jar file: this actually duplicates the main() method, allowing
multiple parse calls and ability to modify options without the overhead of
loading the grammar file each time (and without having to use Java!)
"""

##########################################################################
## Imports
##########################################################################

import sys
import jpype

##########################################################################
## Main Functionality
##########################################################################



def startup(classpath):
    
    '''Start the JVM. This MUST be called before any other jpype functions'''
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=%s" % classpath, "-Xmx500m")




def dictToArgs(d):
    '''Convert a dict of options to a list of command-line-style args'''
    boolDefaults = [ "tokenize", "binarize", "scores", "keepFunctionLabels",
        "substates", "accurate", "modelScore", "confidence", "sentence_likelihood",
        "tree_likelihood", "variational", "render", "chinese", "useGoldPOS",
        "dumpPosteriors", "ec_format",
    ] # these all default to False and only require the switch if True

    # get a list of "-key", "value" or just "-key" if key is in boolDefaults
    args = [j for i in [("-"+k, '%s'%v) if k not in boolDefaults else ("-"+k,) for k,v in iter(d.items())] for j in i]
    return args





def getOpts(args):
    '''
    Converts given command-line-style args to opts for parser functions.

    Note that changing options for:
        accurate, chinese, grFileName, kbest, nGrammars, nThreads, scores,
        substates, viterbi, variational
    after calling loadGrammar will NOT update the parser.

    Specifically, options for:
        grFileName, kbest, nThreads
    are used in both parser setup (loadGrammar) and actual parsing (parseInput)

    Options for:
        binarize, confidence, dumpPosteriors, ec_format, goldPOS, inputFile,
        keepFunctionLabels, maxLength, modelScore, outputFile, render,
        sentence_likelihood, tokenize, tree_likelihood
    do not affect the grammar loading and may be changed between those steps.

    The JVM must be started before calling this function.
    '''
    Options = jpype.JClass("edu.berkeley.nlp.PCFGLA.BerkeleyParser$Options")
    OptionParser = jpype.JClass("edu.berkeley.nlp.PCFGLA.OptionParser")
    optParser = OptionParser(Options)
    opts = optParser.parse(args, True)
    return opts





def loadGrammar(opts):
    '''
    Loads the grammar and lexicon for the parser, given options.
    Returns the initialized parser.
    '''
    threshold = 1.0

    if opts.chinese: #todo WARNING: THIS IS UNTESTED
        Corpus = jpype.JClass("edu.berkeley.nlp.PCFGLA.Corpus")
        Corpus.myTreebank = Corpus.TreeBankType.CHINESE

    parser = None


    if opts.nGrammars != 1: #todo
        print ("Multiple grammars not implemented!")
        sys.exit(1)
    else:
        inFileName = opts.grFileName
        ParserData = jpype.JClass("edu.berkeley.nlp.PCFGLA.ParserData")
        pData = ParserData.Load(inFileName)
        if pData is None:
            print ("Failed to load grammar from file '%s'."%inFileName)
            sys.exit(1)
        grammar = pData.getGrammar()
        lexicon = pData.getLexicon()
        Numberer = jpype.JClass("edu.berkeley.nlp.util.Numberer")
        Numberer.setNumberers(pData.getNumbs())
        if opts.kbest == 1:
            CoarseToFineMaxRuleParser = jpype.JClass("edu.berkeley.nlp.PCFGLA.CoarseToFineMaxRuleParser")
            parser = CoarseToFineMaxRuleParser(grammar, lexicon, threshold, -1,
                opts.viterbi, opts.substates, opts.scores, opts.accurate, opts.variational,
                True, True)
        else:
            CoarseToFineNBestParser = jpype.JClass("edu.berkeley.nlp.PCFGLA.CoarseToFineNBestParser")
            parser = CoarseToFineNBestParser(grammar, lexicon, opts.kbest, threshold,
                -1, opts.viterbi, opts.substates, opts.scores, opts.accurate,
                opts.variational, False, True)

        parser.binarization = pData.getBinarization()


    return parser






    
def parse(parser, opts, sent):
    '''
    Uses parser with opts to parse the input string to output.

    '''

    sent = sent.strip()

    sentence = None

    PTBLineLexer = jpype.JClass("edu.berkeley.nlp.io.PTBLineLexer")
    TreeAnnotations = jpype.JClass("edu.berkeley.nlp.PCFGLA.TreeAnnotations")
    tokenizer = PTBLineLexer()
    sentence = tokenizer.tokenizeLine(sent)

    pt = None
    st = jpype.java.util.ArrayList()
    for s in sentence:
        st.add(s)

    parsedTree = parser.getBestConstrainedParse(st, pt, None)
    parsedTree = TreeAnnotations.unAnnotateTree(parsedTree,opts.keepFunctionLabels)


    return parsedTree




def shutdown():
    '''Shut down the JVM'''
    jpype.shutdownJVM()



