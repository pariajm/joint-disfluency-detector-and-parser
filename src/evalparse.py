"""
Evaluates parse trees with respect to gold trees overall and on
a constituent-by-constituent span basis.  

Also optionally includes word coverage evaluation.

Mark Johnson, 15th November 2018
"""

import collections
import io
import itertools

import tb

Counts = collections.namedtuple('Counts', ('parse', 'gold', 'correct'))
Scores = collections.namedtuple('Scores', ('precision', 'recall', 'fscore'))

def counts_sumcounts(counts, labels=None):
  """Counts should be a CountsType with Counter values for parse, gold 
  and correct, as produced by ParseEval.counts() or ParseEval.wcounts().

  It returns another CountsType with the sum counts, summing over the category
  labels in labels, or over all labels if not specified.
  """
  if labels:
    return Counts(parse=sum(counts.parse.get(label, 0)
                            for label in labels),
                  gold=sum(counts.gold.get(label, 0)
                           for label in labels),
                  correct=sum(counts.correct.get(label, 0)
                              for label in labels))
  else:
    return Counts(parse=sum(counts.parse.values()),
                  gold=sum(counts.gold.values()),
                  correct=sum(counts.correct.values()))


def sumcounts_scores(c):
  """Maps a CountsType with sum counts to a ScoresType (i.e., computes
  precision, recall and f-score).
  """
  return Scores(precision=c.correct/(c.parse+1e-100),
                recall=c.correct/(c.gold+1e-100),
                fscore=2*c.correct/(c.parse+c.gold+2e-100))

class EvalParse:
  """
  This class collects summary statistics for parser evaluation,
  and can print them out in several ways.
  """

  def __init__(self, 
               evaluate_word_coverage=False,
               include_root=True, 
               include_preterminals=False,
               ignore_punctuation=True):
    self.evaluate_word_coverage=evaluate_word_coverage
    self.include_root=include_root
    self.include_preterminals=include_preterminals
    self.ignore_punctuation=ignore_punctuation
    self.reset()


  def reset(self):
    """
    Zeros all counters
    """
    self.parselabel_count = collections.Counter()
    self.goldlabel_count = collections.Counter()
    self.correctlabel_count = collections.Counter()
    
    if self.evaluate_word_coverage:
      self.parselabel_wcount = collections.Counter()
      self.goldlabel_wcount = collections.Counter()
      self.correctlabel_wcount = collections.Counter()


  def update1(self, parse_tree, gold_tree):
    """
    Updates counters based on overlap between
    parse_tree and gold_tree.
    """
    
    if any(pw != gw
           for pw, gw in itertools.zip_longest(tb.tree_terminals(parse_tree),
                                               tb.tree_terminals(gold_tree))):
      raise RuntimeError("parse_tree and gold_tree have different terminal yields:\n"
                         " parse_tree = {}\n"
                         " gold_tree = {}\n".format(parse_tree, gold_tree))

    parse_constituents = tb.tree_constituents(parse_tree,
                                              include_root=self.include_root,
                                              include_preterminals=self.include_preterminals,
                                              ignore_punctuation=self.ignore_punctuation)

    gold_constituents = tb.tree_constituents(gold_tree,
                                             include_root=self.include_root,
                                             include_preterminals=self.include_preterminals,
                                             ignore_punctuation=self.ignore_punctuation)

    parse_constituents = set(parse_constituents)
    gold_constituents = set(gold_constituents)
    correct_constituents = parse_constituents & gold_constituents

    self.parselabel_count.update(c.label for c in parse_constituents)
    self.goldlabel_count.update(c.label for c in gold_constituents)
    self.correctlabel_count.update(c.label for c in correct_constituents)

    if self.evaluate_word_coverage:
      parselabelpos = set((c.label, pos) 
                          for c in parse_constituents
                          for pos in range(c.left, c.right))
      goldlabelpos = set((c.label, pos)
                         for c in gold_constituents
                         for pos in range(c.left, c.right))
      correctlabelpos = parselabelpos & goldlabelpos
      
      self.parselabel_wcount.update(c[0] for c in parselabelpos)
      self.goldlabel_wcount.update(c[0] for c in goldlabelpos)
      self.correctlabel_wcount.update(c[0] for c in correctlabelpos)


  def update(self, parse_trees, gold_trees):
    """
    Updates counters for sequences of parse trees and gold trees.
    """
    for parse_tree, gold_tree in itertools.zip_longest(parse_trees, gold_trees):
      if not parse_tree or not gold_tree:
        raise RuntimeError("parse_trees and gold_trees have different lengths. "
                           "len(parse_trees) = {}, "
                           "len(gold_tree) = {}.".format(len(parse_trees), len(gold_trees)))
      self.update1(parse_tree, gold_tree)


  def __call__(self, parse_trees, gold_trees):
    self.update(parse_trees, gold_trees)


  def counts(self):
    return Counts(parse=self.parselabel_count,
                  gold=self.goldlabel_count,
                  correct=self.correctlabel_count)


  def wcounts(self):
    assert self.evaluate_word_coverage, "Attempting to retrieve word counts from ParseEval(evaluate_word_coverage=False)"
    return Counts(parse=self.parselabel_wcount,
                  gold=self.goldlabel_wcount,
                  correct=self.correctlabel_wcount)


  def scores(self, labels=None):
    """
    Returns the scores for the category labels in labels, or for all
    labels if labels is not specified.
    """
    return sumcounts_scores(counts_sumcounts(self.counts(), labels=labels))


  def wscores(self, labels=None):
    """
    Returns the word scores for the category labels in labels, or for
    all labels if labels is not specified.
    """
    return sumcounts_scores(counts_sumcounts(self.wcounts(), labels=labels))


  def fscore(self, labels=None):
    """Returns the f-score for the category labels in labels, or for all
    labels if labels is not specified.

    """
    return self.scores(labels).fscore


  def wfscore(self, labels=None):
    """Returns the word f-score for the category labels in labels, or for
    all labels if labels is not specified.

    """
    return self.wscores(labels).fscore


  def summary(self, labels=None, wordscores=False):
    """Returns a string containing scores"""
    p = '|'.join(labels)+' ' if labels else ''
    if wordscores:
      p += 'words '
    s = self.wscores(labels) if wordscores else self.scores(labels)
    return ("{0}P: {1.precision:0.4}, "
            "R: {1.recall:0.4}, "
            "F: {1.fscore:0.4}".format(p, s))


  def __str__(self):
    return self.summary()


  def table(self, colsep=",", rowsep="\n", 
            labels=None, extralabels=None,
            summary=True, individual=True,
            float_format="{:.4}"):

    """Returns a string containing a table of results by category label.

    If labels is None, use all category labels in the data.

    If extralabels is not none, write a row for each combination of
    labels in extralabels.

    If individual is True, include a row for each individual label.

    If summary is True, there is a summary row at the end of the table.

    float_format is the format string used to format all floats.
    """

    outf = io.StringIO()

    # write header
    print('label', 'nparse', 'ngold', 'ncorrect', 'precision', 'recall', 'fscore',
          file=outf, sep=colsep, end='')

    if self.evaluate_word_coverage:
      outf.write(colsep)
      print('word nparse','word ngold','word ncorrect','word precision',
            'word recall','word fscore',
            file=outf, sep=colsep, end='')
    outf.write(rowsep)

    def ff(x):  # format float
      return float_format.format(x)  

    def write_row(ls):
      """Writes a row except for the row label"""
      outf.write(colsep)
      sc = counts_sumcounts(self.counts(), labels=ls)
      s = sumcounts_scores(sc)
      print(sc.parse, sc.gold, sc.correct,
            ff(s.precision), ff(s.recall), ff(s.fscore),
            file=outf, sep=colsep, end='')
      if self.evaluate_word_coverage:
        outf.write(colsep)
        sc = counts_sumcounts(self.wcounts(), labels=ls)
        ss = sumcounts_scores(sc)
        print(sc.parse, sc.gold, sc.correct,
              ff(s.precision), ff(s.recall), ff(s.fscore),
              file=outf, sep=colsep, end='')
      outf.write(rowsep)

    if individual:
      if labels:
        ls = labels  # ls is list of labels to iterate through
      else:  # compute ls
        ls = set()
        ls.update(self.parselabel_count.keys(), 
                  self.goldlabel_count.keys(), 
                  self.correctlabel_count.keys())
      for label in ls:
        outf.write(label)
        write_row((label,))
        
    if extralabels:
      for extralabel in extralabels:
        outf.write('|'.join(extralabel))
        write_row(extralabel)

    if summary:
      summarylabel = '|'.join(labels) if labels else "All labels"
      outf.write(summarylabel)
      write_row(labels)

    tbl = outf.getvalue()
    outf.close()
    return tbl


##### Unit tests here

import unittest
import sys

class TestEvalParse(unittest.TestCase):

  def setUp(self):
    gstr = """(S (EDITED (NP (EX there)) (, ,)) 
                 (NP (EX there)) 
                 (VP (BES 's) (NP (DT no) (NN way))) (. .))
              (S (CC and) (, ,) (INTJ (UH uh)) 
                 (PRN (, ,) 
                      (S (NP (PRP you)) (VP (VBP know))) (, ,)) 
                 (NP (DT all))) 
              (S (EDITED (EDITED (EDITED (S (NP (EX There)) (VP (BES 's))) (, ,)) 
                                            (NP (EX there)) (, ,)) (NP (DT th-)) (, ,)) 
                 (NP (DT this) (NN topic)) 
                 (VP (VBZ is) (ADJP (ADVP (RB kind) (RB of)) (TYPO (JJ mute))) (. .) 
                 (INTJ (UH Uh))))
           """
    pstr = """(S (NP (EX there)) 
                 (, ,) 
                 (NP (EX there)) 
                 (VP (BES 's) (NP (DT no) (NN way))) (. .))
              (S1 (CC and) (, ,) (INTJ (UH uh)) (, ,)
                 (PRN (S (NP (PRP you)) (VP (VBP know)))) 
                 (, ,)
                 (NP (DT all))) 
              (S (EDITED (EDITED (EDITED (S (NP (EX There)) (VP (BES 's))) (, ,)) 
                                            (NP (EX there)) (, ,)) (NP (DT th-)) (, ,)) 
                 (NP (DT this) (NN topic)) 
                 (VP (VBZ is) (ADJP (ADVP (RB kind) (RB of)) (TYPO (JJ mute))) (. .) 
                 (INTJ (UH Uh))))
           """
    self.gs = tb.string_trees(gstr)
    self.ps = tb.string_trees(pstr)

    self.e_no_words = EvalParse()
    self.e_no_words(self.ps, self.gs)
    self.e_no_words_tbl = self.e_no_words.table()
    # print(self.e_no_words_tbl)

    self.e_words = EvalParse(evaluate_word_coverage=True)
    self.e_words(self.ps, self.gs)
    self.e_words_tbl = self.e_words.table()
    # print(self.e_words_tbl, file=sys.stderr)
    # print(self.e_no_words)
    # print(self.e_words.summary(labels=('EDITED','PRN','UH')))
    # print(self.e_words.summary(labels=('EDITED','PRN','UH'), wordscores=True))

  def test_label_counts(self):
    self.assertEqual(self.e_no_words.parselabel_count['EDITED'], 3)
    self.assertEqual(self.e_words.goldlabel_wcount['EDITED'], 5)
    self.assertEqual(self.e_words.parselabel_wcount['EDITED'], 4)
    self.assertEqual(self.e_words.parselabel_count['NP'], 9)
    self.assertEqual(self.e_words.parselabel_count['S1'], 1)
    self.assertEqual(self.e_words.goldlabel_count['S1'], 0)
    self.assertEqual(self.e_words.correctlabel_count['S'], 4)
    self.assertEqual(self.e_words.goldlabel_count['S'], 5)


if __name__=='__main__':
  unittest.main()
