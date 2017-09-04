'''
This script returns subgraph vectors for a given set of graphs 
Inputs: 
1) Graph category
2) Number of dimensions
3) Number of negative samples
4) Number of epochs 
5) Batch size
'''
import tensorflow as tf
import glob, os
import numpy as np
import time
import json
import sys
from collections import Counter 

def preprocess(directory):
    target_subgraphs = [] 
    context_subgraphs = []
    target_context_pairs = [] #[(input_tuple,output_tuple)]
    set_of_subgraphs = set()
    all_subgraph_occurance = []
    files = glob.glob(os.path.join(directory,"*.DRemoved"))
    for fil in files:
        with open(fil) as f:
            for line in f:
                line_list = line.split()
                target_subg = line_list[0]
                set_of_subgraphs.add(target_subg)
                context_subgs = line_list[1:]
            #target_context_pairs += [(target_subg,context_subg) for context_subg in context_subgs]
                for context_subg in context_subgs:
                    target_context_pairs.append((target_subg,context_subg))
                    set_of_subgraphs.add(context_subg)
                    all_subgraph_occurance.append(context_subg)
    num_pairs = len(target_context_pairs)
    return (num_pairs, target_context_pairs, all_subgraph_occurance)

def create_lookup_table(subgraphs):
    '''
    Input: entire list of subgraphs
    Output: Dictionary1 id'd by subgraph, and Dictionary2 id'd by integer (so dict2[dict1[subgraph]] = subgraph)
    '''
    sg_counts = Counter(subgraphs)
    sorted_vocab = sorted(sg_counts, key = sg_counts.get, reverse = True) #get subgraphs sorted by frequency
    int_to_vocab = {ii:sgraph for ii, sgraph in enumerate(sorted_vocab)}
    vocab_to_int = {sgraph:ii for ii,sgraph in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


def get_batches_2(batch_size ,vocab_to_int, target_context_pairs):
    '''
    Generator function
    Input: 
        1)Take global variable: target context pairs
        2)Take batch_size
        3)Take global variable: vocab_to_int
        
    Output:
        1)Generator batch object with ([x],[y])
        
    Method: 
        1) at first run of generator, randomly assign indices
        2) Run a ptr through with batch
    1st option given [indices]--> sample [random indices of batch size]
    2nd option given [indices]-->[random indices]--> sample from 0..batch_size-1 then batch_size.... 2*(batch_size)... till remaining_ptr< batch_size
    For given number of pairs, yield batches each of batch_size
    '''
    #Generate n_batches NUMBER of random numbers between 0 to len(tc_pairs)
    #So for n_batches = 5 and len(tc_pairs) = 10 generate 5 random nums between 0 and 10
    total_num_indices = len(target_context_pairs)
    indices = range(total_num_indices)
   # indices = np.random.randint(total_num_indices, size = (total_num_indices))
    np.random.shuffle(indices)
    num_batches = np.ceil(float(total_num_indices)/batch_size)
    
    num_in_last_batch = total_num_indices%batch_size
   
    to_add_to_last_batch = (batch_size - num_in_last_batch)%batch_size
    
    for i in range(int(num_batches)):
        if(i==num_batches-1 and to_add_to_last_batch>0): #last batch
            last_batch_chosen_indices = indices[i*batch_size:] + indices[:to_add_to_last_batch]
            selected_target_context_pairs = np.array(target_context_pairs)[last_batch_chosen_indices]
            
            x_batch_subg = map(lambda pairs: vocab_to_int[pairs[0]], selected_target_context_pairs)
            y_batch_subg = map(lambda pairs: vocab_to_int[pairs[1]], selected_target_context_pairs)
            yield(x_batch_subg,y_batch_subg)
        
        chosen_indices = indices[i*batch_size:(i+1)*(batch_size)] 
        selected_target_context_pairs = np.array(target_context_pairs)[chosen_indices]
        x_batch_subg = map(lambda pairs: vocab_to_int[pairs[0]], selected_target_context_pairs)
        y_batch_subg = map(lambda pairs: vocab_to_int[pairs[1]], selected_target_context_pairs)
        yield(x_batch_subg,y_batch_subg)
        

        
def output_weights_as_json(embeddings_matrix, directory_addr, int_to_vocab):
    get_float_list = lambda x: map(lambda y:float(y),x)
    embeddings = {int_to_vocab[i]:get_float_list(list(embeddings_matrix[i])) for i,_ in enumerate(embeddings_matrix)}
    with open(directory_addr,'w') as f1:
        json.dump(embeddings, f1)
    return embeddings



def g2vec(graph_category, directory, dim, neg_samples, n_epochs, batch_size):
    num_pairs, target_context_pairs, all_subgraph_occurance = preprocess(directory)
    vocab_to_int, int_to_vocab = create_lookup_table(all_subgraph_occurance)
    n_embedding = dim
    epochs = n_epochs
    output_dir = "{}g2v_weights_dims{}_eps{},negsamples{}".format(graph_category, n_embedding, n_epochs, neg_samples)
    '''
    TF initialization code
    '''
    train_graph = tf.Graph() #initialize graph
    n_vocab = len(int_to_vocab)
    
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, [None], name="inputs") #inputs of undefined shape None--> basically a single array of len n_batch
        labels = tf.placeholder(tf.int32, [None, None], name = "labels")
        
        embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding),-1,1))
        embed = tf.nn.embedding_lookup(embedding, inputs) #embedding layer initialization
        
        softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding)))
        softmax_b = tf.Variable(tf.zeros(n_vocab))  #softmax parameter initialization
        
        loss = tf.nn.sampled_softmax_loss(weights= softmax_w, biases = softmax_b, labels = labels, inputs = embed, num_sampled = neg_samples, num_classes = n_vocab) #we have to pass num_classes as labels are integer (not one hot vectors)
        
        cost = tf.reduce_mean(loss) #get loss over all training batch
        optimizer = tf.train.AdamOptimizer().minimize(cost) #optimizer for cost
        
        
    '''
    TF training code
    '''
    with train_graph.as_default():
        saver = tf.train.Saver() #save trained model 
        
    with tf.Session(graph= train_graph) as sess:  #open tf session for graph computation, specify training graph 
        iteration = 1
        loss = 0 
        sess.run(tf.global_variables_initializer()) #initialize all predefined nodes 
        
        for epoch_ in range(1, epochs +1):
            batches = get_batches_2(batch_size, vocab_to_int, target_context_pairs)
            start = time.time()
            
            for x,y in batches:
                feed = {inputs: x, labels: np.array(y)[:, None]} #feed dictionary to placeholders
                train_loss, _ = sess.run([cost, optimizer], feed_dict = feed) #pass cost and optimizer as values to compute
                loss += train_loss
                
                if(iteration%100 == 0):
                    end = time.time()
                    print("Epoch {}/{}".format(epoch_, epochs), "Iteration: {}".format(iteration), "Avg. Training loss: {:.4f}".format(loss/100), "{:.4f}sec/batch".format((end-start)/100))
                    loss = 0
                    start = time.time()
                   
                iteration+=1
        
        
        save_path = saver.save(sess, os.path.join(directory, output_dir+".ckpt"))
        embed_mat = sess.run(embedding) #obtain embedding matrix

        
    '''
    Post Training
    '''
    returned_dict = output_weights_as_json(embed_mat, os.path.join(directory,output_dir +".json"), int_to_vocab)
    print("Written output subgraph embeddings json to {}".format(os.path.join(directory,output_dir+".json")))
        

if __name__ == "__main__":
   # arguments = sys.argv[1:]
    graph_category = sys.argv[1]
    dim = int(sys.argv[2])
    negative_samples = int(sys.argv[3])
    n_epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    if(len(sys.argv)>6):
        root_directory = sys.argv[6]
    else:
        root_directory = "uncompressed_graph_text_files"
        
    directory= os.path.join(root_directory, graph_category)   
    g2vec(graph_category,directory, dim, negative_samples, n_epochs, batch_size)
    
