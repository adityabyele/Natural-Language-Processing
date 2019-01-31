Files:
1 Two Hidden Layers 
  DependencyParser_HL2.py
2 Three Hidden Layers
  DependencyParser_HL3.py
2b Parallel Hidden Layers
   DependencyParser_Parallel.py


Implementation:
1 ParsingSystem.py apply method:
    * Checks if the tree starts with S L or R
    * If the  tree starts with S just Shift
    * If it starts with L add arc from top to second top of the stack and remove second of the stack
    * If it starts with R add arc from second top of the stack to the top and remove the top of the stack

2 DependencyParser.py getFeature method
    * Top 3 of the stack and top 3 of the buffer are added to the list
    * Find the rightmost, leftmost and second rightmost, leftmost of the first two nodes in the list
    * Find the rightmost, leftmost child of the rightmost leftmost child of the top two nodes in the stack
    * Get the Word, POS and the label form the token index and get the correspondng ids.
    * Concat the list and return

3 DependencyParser.py Forward Pass
    * Initialize training inputs as (batchsize, n_tokens), initialize training labels as (batchsize, num of transitions)
    * After getting the embeddings the input is reshaped into (batchsize, n_tokens*embedding size)
    * Initialize weights as normal distribution the size for the hidden layer is (hidd_size, embed_size*n_tokenSize)
    * Initialize weights as normal distribution the size for the output layer is (num of transitions, hidd_size)
    * For the first layer the input is multiplied by the transposed weight. The output is of shape (batch_sz, hidd_sz)
    * For the next layer the out is multiplied by the transposed output weight and the output is of shape(batch_sz, num of transitions)
    * Loss is calculated over this output.

4 DependencyParser.py Loss
    * l2 loss of weights and all embeddings calculated using l2_loss method in tensorflow
    * Crossentropy loss is calculated using softmax loss in tensorflow
    * addition of the above tow losses is the final loss

5 Hidden Layers 2 and 3
    * Same procedure as 3 but the addition of 2/3 more layers more weights need to be added and the shapes modified accordingly 
    * Loss also needs to be added with the extra weights

6 Parallel Layers
    * input is split and more weights with different dimensions are added.







