# Any of the files in glove.6B will work here:

glove_dim = 50
glove_src = os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(glove_dim))
# Creates a dict mapping strings (words) to GloVe vectors:
GLOVE = utils.glove2dict(glove_src)

def glove_vec(w):    
    """Return `w`'s GloVe representation if available, else return 
    a random vector."""
    return GLOVE.get(w, randvec(w, n=glove_dim))


def vec_concatenate(u, v):
    return np.concatenate((u, v))

def vec_diff(u, v):
    return np.subtract(u, v)
    
def vec_max(u, v):
    return np.maximum(u,v)

#normal default setting 
net = TorchShallowNeuralClassifier(hidden_dim=50, max_iter=100)


import torch.nn as nn

class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self, dropout_prob=0.7, **kwargs):
        self.dropout_prob = dropout_prob
        super().__init__(**kwargs)
    
    def define_graph(self):
        """Complete this method!
        
        Returns
        -------
        an `nn.Module` instance, which can be a free-standing class you 
        write yourself, as in `torch_rnn_classifier`, or the outpiut of 
        `nn.Sequential`, as in `torch_shallow_neural_classifier`.
        
        """
        ##### YOUR CODE HERE
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(self.dropout_prob),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes_))



from torch_rnn_classifier import TorchRNNClassifier
vocab = utils.get_vocab(X, n_words=10000)
mod = TorchRNNClassifier(vocab, hidden_dim=50, max_iter=10)




word_disjoint_experiment = nli.wordentail_experiment(
    train_data=wordentail_data['word_disjoint']['train'],
    assess_data=wordentail_data['word_disjoint']['dev'], 
    model=net, 
    vector_func=glove_vec,
    vector_combo_func=vec_concatenate)

word_experiment = nli.wordentail_experiment(
                train_data=wordentail_data[data]['train'],
                assess_data=wordentail_data[data]['dev'], 
                model=model, 
                vector_func=glove_vec,
                vector_combo_func=v_func)

wordentail_data.keys()
wordentail_data['edge_disjoint'].keys()

