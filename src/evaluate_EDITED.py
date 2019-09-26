"""
Evaluate f-score of all nodes (like EVALB), and EDITED and INTJ words

(c) Mark Johnson, 15th November 2018
"""

import itertools

import trees

import evalparse
import tb

def tree_tb(tree):
    """
    Map a trees.InternalTreebankNode to a tb tree.
    This is based on trees.InternalTreebankNode.linearize()
    """
    assert isinstance(tree, trees.InternalTreebankNode) or isinstance(tree, trees.LeafTreebankNode)
    if isinstance(tree, trees.InternalTreebankNode):
        return [tree.label] + list(map(tree_tb, tree.children))
    elif isinstance(tree, trees.LeafTreebankNode):
        return [tree.tag, tree.word]
    else:
        assert False, "tree is not a trees.InternalTreebankNode or a trees.LeafTreebankNode"

class Evaluate:
    """Evaluation for the self-attentive parser.
    """
    def __init__(self, gold_trees, predicted_trees, elabels=('EDITED',), dlabels=('EDITED','PRN','UH')):
        self.dlabels = dlabels
        self.elabels = elabels
        assert len(gold_trees) == len(predicted_trees)
        e = evalparse.EvalParse(evaluate_word_coverage=True)
        for g, p in zip(gold_trees, predicted_trees):
            e.update1(tree_tb(g), tree_tb(p))
        self.evalparse = e
        self.fscore = e.fscore()
        self.efscore = e.fscore(labels=elabels)

    def __str__(self):
        return '; '.join((self.evalparse.summary(), 
                          self.evalparse.summary(labels=self.elabels), 
                          self.evalparse.summary(labels=self.dlabels),
                          self.evalparse.summary(labels=self.elabels, wordscores=True), 
                          self.evalparse.summary(labels=self.dlabels, wordscores=True)))

    def table(self):
        return self.evalparse.table(extralabels=(self.dlabels,))
