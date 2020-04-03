#!/usr/bin/env python
# coding: utf-8

# # Homework and bake-off: Word similarity

# In[ ]:


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


# ## Contents
# 
# 1. [Overview](#Overview)
# 1. [Set-up](#Set-up)
# 1. [Dataset readers](#Dataset-readers)
# 1. [Dataset comparisons](#Dataset-comparisons)
#   1. [Vocab overlap](#Vocab-overlap)
#   1. [Pair overlap and score correlations](#Pair-overlap-and-score-correlations)
# 1. [Evaluation](#Evaluation)
#   1. [Dataset evaluation](#Dataset-evaluation)
#   1. [Dataset error analysis](#Dataset-error-analysis)
#   1. [Full evaluation](#Full-evaluation)
# 1. [Homework questions](#Homework-questions)
#   1. [PPMI as a baseline [0.5 points]](#PPMI-as-a-baseline-[0.5-points])
#   1. [Gigaword with LSA at different dimensions [0.5 points]](#Gigaword-with-LSA-at-different-dimensions-[0.5-points])
#   1. [Gigaword with GloVe for a small number of iterations [0.5 points]](#Gigaword-with-GloVe-for-a-small-number-of-iterations-[0.5-points])
#   1. [Dice coefficient [0.5 points]](#Dice-coefficient-[0.5-points])
#   1. [t-test reweighting [2 points]](#t-test-reweighting-[2-points])
#   1. [Enriching a VSM with subword information [2 points]](#Enriching-a-VSM-with-subword-information-[2-points])
#   1. [Your original system [3 points]](#Your-original-system-[3-points])
# 1. [Bake-off [1 point]](#Bake-off-[1-point])

# ## Overview
# 
# Word similarity datasets have long been used to evaluate distributed representations. This notebook provides basic code for conducting such analyses with a number of datasets:
# 
# | Dataset | Pairs | Task-type | Current best Spearman $\rho$ | Best $\rho$ paper |   |
# |---------|-------|-----------|------------------------------|-------------------|---|
# | [WordSim-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) | 353 | Relatedness | 82.8 | [Speer et al. 2017](https://arxiv.org/abs/1612.03975) |
# | [MTurk-771](http://www2.mta.ac.il/~gideon/mturk771.html) | 771 | Relatedness | 81.0 | [Speer et al. 2017](https://arxiv.org/abs/1612.03975) |
# | [The MEN Test Collection](http://clic.cimec.unitn.it/~elia.bruni/MEN) | 3,000 | Relatedness | 86.6 | [Speer et al. 2017](https://arxiv.org/abs/1612.03975)  | 
# | [SimVerb-3500-dev](http://people.ds.cam.ac.uk/dsg40/simverb.html) | 500 | Similarity | 61.1 | [Mrki&scaron;&cacute; et al. 2016](https://arxiv.org/pdf/1603.00892.pdf) |
# | [SimVerb-3500-test](http://people.ds.cam.ac.uk/dsg40/simverb.html) | 3,000 | Similarity | 62.4 | [Mrki&scaron;&cacute; et al. 2016](https://arxiv.org/pdf/1603.00892.pdf) |
# 
# Each of the similarity datasets contains word pairs with an associated human-annotated similarity score. (We convert these to distances to align intuitively with our distance measure functions.) The evaluation code measures the distance between the word pairs in your chosen VSM (which should be a `pd.DataFrame`).
# 
# The evaluation metric for each dataset is the [Spearman correlation coefficient $\rho$](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) between the annotated scores and your distances, as is standard in the literature. We also macro-average these correlations across the datasets for an overall summary. (In using the macro-average, we are saying that we care about all the datasets equally, even though they vary in size.)
# 
# This homework ([questions at the bottom of this notebook](#Homework-questions)) asks you to write code that uses the count matrices in `data/vsmdata` to create and evaluate some baseline models as well as an original model $M$ that you design. This accounts for 9 of the 10 points for this assignment.
# 
# For the associated bake-off, we will distribute two new word similarity or relatedness datasets and associated reader code, and you will evaluate $M$ (no additional training or tuning allowed!) on those new datasets. Systems that enter will receive the additional homework point, and systems that achieve the top score will receive an additional 0.5 points.

# ## Set-up

# In[ ]:


from collections import defaultdict
import csv
import itertools
import numpy as np
import os
import pandas as pd
from scipy.stats import spearmanr
import vsm
from IPython.display import display


# In[ ]:


VSM_HOME = os.path.join('data', 'vsmdata')

WORDSIM_HOME = os.path.join('data', 'wordsim')


# In[ ]:


VSM_HOME


# ## Dataset readers

# In[ ]:


def wordsim_dataset_reader(
        src_filename, 
        header=False, 
        delimiter=',', 
        score_col_index=2):
    """Basic reader that works for all similarity datasets. They are 
    all tabular-style releases where the first two columns give the 
    word and a later column (`score_col_index`) gives the score.

    Parameters
    ----------
    src_filename : str
        Full path to the source file.
    header : bool
        Whether `src_filename` has a header. Default: False
    delimiter : str
        Field delimiter in `src_filename`. Default: ','
    score_col_index : int
        Column containing the similarity scores Default: 2

    Yields
    ------
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity
       score in the file so that we are intuitively aligned with our
       distance-based code. To align with our VSMs, all the words are 
       downcased.

    """
    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)
        for row in reader:
            w1 = row[0].strip().lower()
            w2 = row[1].strip().lower()
            score = row[score_col_index]
            # Negative of scores to align intuitively with distance functions:
            score = -float(score)
            yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'wordsim353', 'combined.csv')
    return wordsim_dataset_reader(
        src_filename, header=True)

def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'MTURK-771.csv')
    return wordsim_dataset_reader(
        src_filename, header=False)

def simverb3500dev_reader():
    """SimVerb-3500: http://people.ds.cam.ac.uk/dsg40/simverb.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'SimVerb-3500', 'SimVerb-500-dev.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=True, score_col_index=3)

def simverb3500test_reader():
    """SimVerb-3500: http://people.ds.cam.ac.uk/dsg40/simverb.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'SimVerb-3500', 'SimVerb-3000-test.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=True, score_col_index=3)

def men_reader():
    """MEN: http://clic.cimec.unitn.it/~elia.bruni/MEN"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'MEN', 'MEN_dataset_natural_form_full')
    return wordsim_dataset_reader(
        src_filename, header=False, delimiter=' ') 


# This collection of readers will be useful for flexible evaluations:

# In[ ]:


READERS = (wordsim353_reader, mturk771_reader, simverb3500dev_reader, 
           simverb3500test_reader, men_reader)


# ## Dataset comparisons
# 
# This section does some basic analysis of the datasets. The goal is to obtain a deeper understanding of what problem we're solving – what strengths and weaknesses the datasets have and how they relate to each other. For a full-fledged project, we would want to continue work like this and report on it in the paper, to provide context for the results.

# In[ ]:


def get_reader_name(reader):
    """Return a cleaned-up name for the similarity dataset 
    iterator `reader`
    """
    return reader.__name__.replace("_reader", "")


# ### Vocab overlap
# 
# How many vocabulary items are shared across the datasets?

# In[ ]:


def get_reader_vocab(reader):
    """Return the set of words (str) in `reader`."""
    vocab = set()
    for w1, w2, _ in reader():
        vocab.add(w1)
        vocab.add(w2)
    return vocab


# In[ ]:


def get_reader_vocab_overlap(readers=READERS):
    """Get data on the vocab-level relationships between pairs of 
    readers. Returns a a pd.DataFrame containing this information.
    """
    data = []
    for r1, r2 in itertools.product(readers, repeat=2):       
        v1 = get_reader_vocab(r1)
        v2 = get_reader_vocab(r2)
        d = {
            'd1': get_reader_name(r1),
            'd2': get_reader_name(r2),
            'overlap': len(v1 & v2), 
            'union': len(v1 | v2),
            'd1_size': len(v1),
            'd2_size': len(v2)}
        data.append(d)
    return pd.DataFrame(data)


# In[ ]:


vocab_overlap = get_reader_vocab_overlap()


# In[ ]:


def vocab_overlap_crosstab(vocab_overlap):
    """Return an intuitively formatted `pd.DataFrame` giving 
    vocab-overlap counts for all the datasets represented in 
    `vocab_overlap`, the output of `get_reader_vocab_overlap`.
    """        
    xtab = pd.crosstab(
        vocab_overlap['d1'], 
        vocab_overlap['d2'], 
        values=vocab_overlap['overlap'], 
        aggfunc=np.mean)
    # Blank out the upper right to reduce visual clutter:
    for i in range(0, xtab.shape[0]):
        for j in range(i+1, xtab.shape[1]):
            xtab.iloc[i, j] = ''        
    return xtab        


# In[ ]:


vocab_overlap_crosstab(vocab_overlap)


# This looks reasonable. By design, the SimVerb dev and test sets have a lot of overlap. The other overlap numbers are pretty small, even adjusting for dataset size.

# ### Pair overlap and score correlations
# 
# How many word pairs are shared across datasets and, for shared pairs, what is the correlation between their scores? That is, do the datasets agree?

# In[ ]:


def get_reader_pairs(reader):
    """Return the set of alphabetically-sorted word (str) tuples 
    in `reader`
    """
    return {tuple(sorted([w1, w2])): score for w1, w2, score in reader()}


# In[ ]:


def get_reader_pair_overlap(readers=READERS):
    """Return a `pd.DataFrame` giving the number of overlapping 
    word-pairs in pairs of readers, along with the Spearman 
    correlations.
    """    
    data = []
    for r1, r2 in itertools.product(READERS, repeat=2):
        if r1.__name__ != r2.__name__:
            d1 = get_reader_pairs(r1)
            d2 = get_reader_pairs(r2)
            overlap = []
            for p, s in d1.items():
                if p in d2:
                    overlap.append([s, d2[p]])
            if overlap:
                s1, s2 = zip(*overlap)
                rho = spearmanr(s1, s2)[0]
            else:
                rho = None
            # Canonical order for the pair:
            n1, n2 = sorted([get_reader_name(r1), get_reader_name(r2)])
            d = {
                'd1': n1,
                'd2': n2,
                'pair_overlap': len(overlap),
                'rho': rho}
            data.append(d)
    df = pd.DataFrame(data)
    df = df.sort_values(['pair_overlap','d1','d2'], ascending=False)
    # Return only every other row to avoid repeats:
    return df[::2].reset_index(drop=True)


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    display(get_reader_pair_overlap())


# This looks reasonable: none of the datasets have a lot of overlapping pairs, so we don't have to worry too much about places where they give conflicting scores.

# ## Evaluation
# 
# This section builds up the evaluation code that you'll use for the homework and bake-off. For illustrations, I'll read in a VSM created from `data/vsmdata/giga_window5-scaled.csv.gz`:

# In[ ]:


giga5 = pd.read_csv(
    os.path.join(VSM_HOME, "giga_window5-scaled.csv.gz"), index_col=0)


# In[ ]:


giga5.head()


# In[ ]:


giga5.shape


# ### Dataset evaluation

# In[ ]:


def word_similarity_evaluation(reader, df, distfunc=vsm.cosine):
    """Word-similarity evalution framework.
    
    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).    
    df : pd.DataFrame
        The VSM being evaluated.        
    distfunc : function mapping vector pairs to floats.
        The measure of distance between vectors. Can also be 
        `vsm.euclidean`, `vsm.matching`, `vsm.jaccard`, as well as 
        any other float-valued function on pairs of vectors.    
        
    Raises
    ------
    ValueError
        If `df.index` is not a subset of the words in `reader`.
    
    Returns
    -------
    float, data
        `float` is the Spearman rank correlation coefficient between 
        the dataset scores and the similarity values obtained from 
        `df` using  `distfunc`. This evaluation is sensitive only to 
        rankings, not to absolute values.  `data` is a `pd.DataFrame` 
        with columns['word1', 'word2', 'score', 'distance'].
        
    """
    data = []
    for w1, w2, score in reader():
        d = {'word1': w1, 'word2': w2, 'score': score}
        for w in [w1, w2]:
            if w not in df.index:
                raise ValueError(
                    "Word '{}' is in the similarity dataset {} but not in the "
                    "DataFrame, making this evaluation ill-defined. Please "
                    "switch to a DataFrame with an appropriate vocabulary.".
                    format(w, get_reader_name(reader))) 
        d['distance'] = distfunc(df.loc[w1], df.loc[w2])
        data.append(d)
    data = pd.DataFrame(data)
    rho, pvalue = spearmanr(data['score'].values, data['distance'].values)
    return rho, data


# In[ ]:


rho, eval_df = word_similarity_evaluation(men_reader, giga5)


# In[ ]:


rho


# In[ ]:


eval_df.head()


# ### Dataset error analysis
# 
# For error analysis, we can look at the words with the largest delta between the gold score and the distance value in our VSM. We do these comparisons based on ranks, just as with our primary metric (Spearman $\rho$), and we normalize both rankings so that they have a comparable number of levels.

# In[ ]:


def word_similarity_error_analysis(eval_df):    
    eval_df['distance_rank'] = _normalized_ranking(eval_df['distance'])
    eval_df['score_rank'] = _normalized_ranking(eval_df['score'])
    eval_df['error'] =  abs(eval_df['distance_rank'] - eval_df['score_rank'])
    return eval_df.sort_values('error')
    
    
def _normalized_ranking(series):
    ranks = series.rank(method='dense')
    return ranks / ranks.sum()    


# Best predictions:

# In[ ]:


word_similarity_error_analysis(eval_df).head()


# Worst predictions:

# In[ ]:


word_similarity_error_analysis(eval_df).tail()


# ### Full evaluation

# A full evaluation is just a loop over all the readers on which one want to evaluate, with a macro-average at the end:

# In[ ]:


def full_word_similarity_evaluation(df, readers=READERS, distfunc=vsm.cosine):
    """Evaluate a VSM against all datasets in `readers`.
    
    Parameters
    ----------
    df : pd.DataFrame
    readers : tuple 
        The similarity dataset readers on which to evaluate.
    distfunc : function mapping vector pairs to floats.
        The measure of distance between vectors. Can also be 
        `vsm.euclidean`, `vsm.matching`, `vsm.jaccard`, as well as 
        any other float-valued function on pairs of vectors.    
    
    Returns
    -------
    pd.Series
        Mapping dataset names to Spearman r values.
        
    """        
    scores = {}     
    for reader in readers:
        score, data_df = word_similarity_evaluation(reader, df, distfunc=distfunc)
        scores[get_reader_name(reader)] = score
    series = pd.Series(scores, name='Spearman r')
    series['Macro-average'] = series.mean()
    return series


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    display(full_word_similarity_evaluation(giga5))


# ## Homework questions
# 
# Please embed your homework responses in this notebook, and do not delete any cells from the notebook. (You are free to add as many cells as you like as part of your responses.)

# ### PPMI as a baseline [0.5 points]
# 
# The insight behind PPMI is a recurring theme in word representation learning, so it is a natural baseline for our task. For this question, write a function called `run_giga_ppmi_baseline` that does the following:
# 
# 1. Reads the Gigaword count matrix with a window of 20 and a flat scaling function into a `pd.DataFrame`s, as is done in the VSM notebooks. The file is `data/vsmdata/giga_window20-flat.csv.gz`, and the VSM notebooks provide examples of the needed code.
# 
# 1. Reweights this count matrix with PPMI.
# 
# 1. Evaluates this reweighted matrix using `full_word_similarity_evaluation`. The return value of `run_giga_ppmi_baseline` should be the return value of this call to `full_word_similarity_evaluation`.
# 
# The goal of this question is to help you get more familiar with the code in `vsm` and the function `full_word_similarity_evaluation`.
# 
# The function `test_run_giga_ppmi_baseline` can be used to test that you've implemented this specification correctly.

# In[ ]:


def run_giga_ppmi_baseline():
    ##### YOUR CODE HERE
    giga20 = pd.read_csv(os.path.join(VSM_HOME, 'giga_window20-flat.csv.gz'), index_col=0)
    giga20_ppmi_reweights = vsm.pmi(giga20, positive=True)
    result = full_word_similarity_evaluation(giga20_ppmi_reweights)
    return result 


# In[ ]:


def test_run_giga_ppmi_baseline(run_giga_ppmi_baseline):
    result = run_giga_ppmi_baseline()
    ws_result = result.loc['wordsim353'].round(2)
    ws_expected = 0.58
    assert ws_result == ws_expected,         "Expected wordsim353 value of {}; got {}".format(ws_expected, ws_result)


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_giga_ppmi_baseline(run_giga_ppmi_baseline)


# ### Gigaword with LSA at different dimensions [0.5 points]
# 
# We might expect PPMI and LSA to form a solid pipeline that combines the strengths of PPMI with those of dimensionality reduction. However, LSA has a hyper-parameter $k$ – the dimensionality of the final representations – that will impact performance. For this problem, write a wrapper function `run_ppmi_lsa_pipeline` that does the following:
# 
# 1. Takes as input a count `pd.DataFrame` and an LSA parameter `k`.
# 1. Reweights the count matrix with PPMI.
# 1. Applies LSA with dimensionality `k`.
# 1. Evaluates this reweighted matrix using `full_word_similarity_evaluation`. The return value of `run_ppmi_lsa_pipeline` should be the return value of this call to `full_word_similarity_evaluation`.
# 
# The goal of this question is to help you get a feel for how much LSA alone can contribute to this problem. 
# 
# The  function `test_run_ppmi_lsa_pipeline` will test your function on the count matrix in `data/vsmdata/giga_window20-flat.csv.gz`.

# In[ ]:


def run_ppmi_lsa_pipeline(count_df, k):
    ##### YOUR CODE HERE
    ppmi_reweight_df = vsm.pmi(count_df)
    lsa_df = vsm.lsa(ppmi_reweight_df, k)
    results = full_word_similarity_evaluation(lsa_df)
    return results 
    


# In[ ]:


def test_run_ppmi_lsa_pipeline(run_ppmi_lsa_pipeline):
    giga20 = pd.read_csv(
        os.path.join(VSM_HOME, "giga_window20-flat.csv.gz"), index_col=0)
    results = run_ppmi_lsa_pipeline(giga20, k=10)
    men_expected = 0.57
    men_result = results.loc['men'].round(2)
    assert men_result == men_expected,        "Expected men value of {}; got {}".format(men_expected, men_result)


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_ppmi_lsa_pipeline(run_ppmi_lsa_pipeline)


# ### Gigaword with GloVe for a small number of iterations [0.5 points]
# 
# Ideally, we would run GloVe for a very large number of iterations on a GPU machine to compare it against its close cousin PMI. However, we don't want this homework to cost you a lot of money or monopolize a lot of your available computing resources, so let's instead just probe GloVe a little bit to see if it has promise for our task. For this problem, write a function `run_small_glove_evals` that does the following:
# 
# 1. Reads in `data/vsmdata/giga_window20-flat.csv.gz`.
# 1. Runs GloVe for 10, 100, and 200 iterations on `data/vsmdata/giga_window20-flat.csv.gz`, using the `mittens` implementation of `GloVe`. 
#   * For all the other parameters to `mittens.GloVe` besides `max_iter`, use the package's defaults.
#   * Because of the way that implementation is designed, these will have to be separate runs, but they should be relatively quick. 
# 1. Stores the values in a `dict` mapping each `max_iter` value to its associated 'Macro-average' score according to `full_word_similarity_evaluation`. `run_small_glove_evals`  should return this `dict`.
# 
# The trend should give you a sense for whether it is worth running GloVe for more iterations.
# 
# Some implementation notes:
# 
# * Your trained GloVe matrix `X` needs to be wrapped in a `pd.DataFrame` to work with `full_word_similarity_evaluation`. `pd.DataFrame(X, index=giga20.index)` will do the trick.
# 
# * If `glv` is your GloVe model, then running `glv.sess.close()` after each model is trained will silence warnings from TensorFlow about interactive sessions being active.
# 
# Performance will vary a lot for this function, so there is some uncertainty in the testing, but `test_run_small_glove_evals` will at least check that you wrote a function with the right general logic.

# In[ ]:


def run_small_glove_evals():

    from mittens import GloVe
    ##### YOUR CODE HERE
    giga20 = pd.read_csv(os.path.join(VSM_HOME, 'giga_window20-flat.csv.gz'), index_col=0)
    result = {} 
    for i in [10, 100, 200]:
        glv = GloVe(max_iter=i)
        X = glv.fit(giga20)
        glv_df = pd.DataFrame(X, index=giga20.index)
        evaluate = full_word_similarity_evaluation(glv_df)
        result[i] =  evaluate['Macro-average']
    glv.sess.close()
    return result 
    


# In[ ]:


def test_run_small_glove_evals(run_small_glove_evals):
    data = run_small_glove_evals()
    for max_iter in (10, 100, 200):
        assert max_iter in data
        assert isinstance(data[max_iter], float)


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_small_glove_evals(run_small_glove_evals)


# ### Dice coefficient [0.5 points]
# 
# Implement the Dice coefficient for real-valued vectors, as
# 
# $$
# \textbf{dice}(u, v) = 
# 1 - \frac{
#   2 \sum_{i=1}^{n}\min(u_{i}, v_{i})
# }{
#     \sum_{i=1}^{n} u_{i} + v_{i}
# }$$
#  
# You can use `test_dice_implementation` below to check that your implementation is correct.

# In[ ]:


def test_dice_implementation(func):
    """`func` should be an implementation of `dice` as defined above."""
    X = np.array([
        [  4.,   4.,   2.,   0.],
        [  4.,  61.,   8.,  18.],
        [  2.,   8.,  10.,   0.],
        [  0.,  18.,   0.,   5.]]) 
    assert func(X[0], X[1]).round(5) == 0.80198
    assert func(X[1], X[2]).round(5) == 0.67568


# In[ ]:


def dice(u, v):
    
    ##### YOUR CODE HERE
    return 1.0 - ( 2*np.sum(np.minimum(u, v))/ sum(u+v))


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_dice_implementation(dice)


# ### t-test reweighting [2 points]
# 
# 

# The t-test statistic can be thought of as a reweighting scheme. For a count matrix $X$, row index $i$, and column index $j$:
# 
# $$\textbf{ttest}(X, i, j) = 
# \frac{
#     P(X, i, j) - \big(P(X, i, *)P(X, *, j)\big)
# }{
# \sqrt{(P(X, i, *)P(X, *, j))}
# }$$
# 
# where $P(X, i, j)$ is $X_{ij}$ divided by the total values in $X$, $P(X, i, *)$ is the sum of the values in row $i$ of $X$ divided by the total values in $X$, and $P(X, *, j)$ is the sum of the values in column $j$ of $X$ divided by the total values in $X$.
# 
# For this problem, implement this reweighting scheme. You can use `test_ttest_implementation` below to check that your implementation is correct. You do not need to use this for any evaluations, though we hope you will be curious enough to do so!

# In[ ]:


def test_ttest_implementation(func):
    """`func` should be an implementation of t-test reweighting as 
    defined above.
    """
    X = pd.DataFrame(np.array([
        [  4.,   4.,   2.,   0.],
        [  4.,  61.,   8.,  18.],
        [  2.,   8.,  10.,   0.],
        [  0.,  18.,   0.,   5.]]))    
    actual = np.array([
        [ 0.33056, -0.07689,  0.04321, -0.10532],
        [-0.07689,  0.03839, -0.10874,  0.07574],
        [ 0.04321, -0.10874,  0.36111, -0.14894],
        [-0.10532,  0.07574, -0.14894,  0.05767]])    
    predicted = func(X)
    assert np.array_equal(predicted.round(5), actual)


# In[ ]:


def ttest(df):
    
    ##### YOUR CODE HERE - revise 
    df = df/df.sum().sum()
    col_total = df.sum(axis=0)
    row_total = df.sum(axis=1)
    
    m = np.outer(row_total, col_total)
   
    numerator = df - m
    denominator = m**0.5
    return numerator/denominator


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_ttest_implementation(ttest)


# ### Enriching a VSM with subword information [2 points]
# 
# It might be useful to combine character-level information with word-level information. To help you begin asssessing this idea, this question asks you to write a function that modifies an existing VSM so that the representation for each word $w$ is the element-wise sum of $w$'s original word-level representation with all the representations for the n-grams $w$ contains. 
# 
# The following starter code should help you structure this and clarify the requirements, and a simple test is included below as well.
# 
# You don't need to write a lot of code; the motivation for this question is that the function you write could have practical value.

# In[ ]:


def subword_enrichment(df, n=4):
    
    # 1. Use `vsm.ngram_vsm` to create a character-level 
    # VSM from `df`, using the above parameter `n` to 
    # set the size of the ngrams.
    
    ##### YOUR CODE HERE
    df_ngrams = vsm.ngram_vsm(df, n)
        
    # 2. Use `vsm.character_level_rep` to get the representation
    # for every word in `df` according to the character-level
    # VSM you created above.
    
    ##### YOUR CODE HERE
    new_matrix = []
    for i in df.index:
        new_vec = vsm.character_level_rep(i, df_ngrams, n)
        new_matrix.append(new_vec) 

    
    # 3. For each representation created at step 2, add in its
    # original representation from `df`. (This should use
    # element-wise addition; the dimensionality of the vectors
    # will be unchanged.)
                            
    ##### YOUR CODE HERE
    df_sub = pd.DataFrame(new_matrix, df.index)
    df_sub_sum = df.add(df_sub)
    
    
    # 4. Return a `pd.DataFrame` with the same index and column
    # values as `df`, but filled with the new representations
    # created at step 3.
                            
    ##### YOUR CODE HERE
    return df_sub_sum


# In[ ]:


def test_subword_enrichment(func):
    """`func` should be an implementation of subword_enrichment as 
    defined above.
    """
    vocab = ["ABCD", "BCDA", "CDAB", "DABC"]
    df = pd.DataFrame([
        [1, 1, 2, 1],
        [3, 4, 2, 4],
        [0, 0, 1, 0],
        [1, 0, 0, 0]], index=vocab)
    expected = pd.DataFrame([
        [14, 14, 18, 14],
        [22, 26, 18, 26],
        [10, 10, 14, 10],
        [14, 10, 10, 10]], index=vocab)
    new_df = func(df, n=2)
    assert np.array_equal(expected.columns, new_df.columns),         "Columns are not the same"
    assert np.array_equal(expected.index, new_df.index),         "Indices are not the same"
    assert np.array_equal(expected.values, new_df.values),         "Co-occurrence values aren't the same"    


# In[ ]:


if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_subword_enrichment(subword_enrichment)


# ### Your original system [3 points]
# 
# This question asks you to design your own model. You can of course include steps made above (ideally, the above questions informed your system design!), but your model should not be literally identical to any of the above models. Other ideas: retrofitting, autoencoders, GloVe, subword modeling, ... 
# 
# Requirements:
# 
# 1. Your code must operate on one of the count matrices in `data/vsmdata`. You can choose which one. __Other pretrained vectors cannot be introduced__.
# 
# 1. Your code must be self-contained, so that we can work with your model directly in your homework submission notebook. If your model depends on external data or other resources, please submit a ZIP archive containing these resources along with your submission.
# 
# In the cell below, please provide a brief technical description of your original system, so that the teaching team can gain an understanding of what it does. This will help us to understand your code and analyze all the submissions to identify patterns and strategies.

# In[ ]:


# Enter your system description in this cell.
# Please do not remove this comment.
if 'IS_GRADESCOPE_ENV' not in os.environ:
    pass
    #Due to the time constraint, re-designing matrix for encoding or creating new features for each word-pair 
    #has been considered but not implemented. 
    #Instead, some quick test on the more robust model for example GloVe and autoencoders are implemented 
    
    #The critical of getting a good result are:
    # 1, Applying proper re-weighting mechanism (lack of statstical analysis, currently implementing two algo based on best result test on local)
    # 2, Test on didmention reduction (this is optional because of the for certain model, 
    #    parse matrixs might contain granular informaiton)
    # 3, Given the fact that the new system is dependence on general model like GloVe and autoencoders, funing 
    #    the parameters dose give the system a boost in accuracy 
    


# ## Bake-off [1 point]
# 
# For the bake-off, we will release two additional datasets. The announcement will go out on the discussion forum. We will also release reader code for these datasets that you can paste into this notebook. You will evaluate your custom model $M$ (from the previous question) on these new datasets using `full_word_similarity_evaluation`. Rules:
# 
# 1. Only one evaluation is permitted.
# 1. No additional system tuning is permitted once the bake-off has started.
# 
# The cells below this one constitute your bake-off entry.
# 
# People who enter will receive the additional homework point, and people whose systems achieve the top score will receive an additional 0.5 points. We will test the top-performing systems ourselves, and only systems for which we can reproduce the reported results will win the extra 0.5 points.
# 
# Late entries will be accepted, but they cannot earn the extra 0.5 points. Similarly, you cannot win the bake-off unless your homework is submitted on time.
# 
# The announcement will include the details on where to submit your entry.

# In[ ]:


# Enter your bake-off assessment code into this cell. 
# Please do not remove this comment.

if 'IS_GRADESCOPE_ENV' not in os.environ:
    pass
    


# In[ ]:


# On an otherwise blank line in this cell, please enter
# your "Macro-average" value as reported by the code above. 
# Please enter only a number between 0 and 1 inclusive.
# Please do not remove this comment.
if 'IS_GRADESCOPE_ENV' not in os.environ:
    pass
    # Please enter your score in the scope of the above conditional.
    ##### YOUR CODE HERE
   
    


