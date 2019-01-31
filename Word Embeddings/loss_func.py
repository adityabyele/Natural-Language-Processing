import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================    
    """
    t0 = tf.transpose(true_w)
    t1 = tf.matmul(inputs, t0)
    A = tf.log(tf.exp(tf.diag_part(t1)))    

    print(A.shape)
    
    B = tf.transpose(tf.log(tf.reduce_sum(tf.exp(t1), axis=0)))
    print(B.shape)


    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    

    labels = tf.reshape(labels, [-1])    
    u0 = tf.gather(weights, labels)    
    vk = tf.gather(weights, sample)    
    probs = tf.convert_to_tensor(unigram_prob)
    probs_u0 = tf.gather(probs, labels)
    probs_vk = tf.gather(probs, sample)
    biases_u0 = tf.gather(biases, labels)
    biases_vk = tf.gather(biases, sample)
    k = sample.shape[0]
    vc = inputs

    print("u0", u0.shape)
    print("vk", vk.shape)
    print("probs_u0", probs_u0.shape)
    print("probs_vk", probs_vk.shape)
    print("biases_u0", biases_u0.shape)
    print("biases_vk", biases_vk.shape)
    print("k", k)
    # print("probs", probs.shape)
    # print("inputs", inputs.shape)
    # print("weights", weights.shape)
    # print("biases", biases.shape)
    # print("labels", labels.shape)
    # print("sample", sample.shape)
    # print("unigram prob", len(unigram_prob))

    u0 = tf.transpose(u0)
    vu = tf.matmul(vc, u0)
    vu = tf.diag_part(vu)
    vub = tf.add(vu, biases_u0)
    lgP0 = tf.log(tf.scalar_mul(k, probs_u0) +1e-10)
    A = tf.log(tf.sigmoid(tf.subtract(vub, lgP0))+1e-10)


    vk = tf.transpose(vk)
    vvk = tf.matmul(vc, vk)
    vvkb = tf.add(vvk, biases_vk)
    lgPx = tf.log(tf.scalar_mul(k, probs_vk)+1e-10)
    lgPwx = tf.log(tf.subtract(1.0, tf.sigmoid(tf.subtract(vvkb, lgPx)))+1e-10)
    B = tf.transpose(tf.reduce_sum(lgPwx, axis = 1))

    C = tf.scalar_mul(-1, tf.add(A,B))
    print("C", C.shape)

    return C