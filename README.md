# Benchmarking Autoencoders

The basic model here implements a Denoising Autoencoder on the MNIST dataset ([Theory](#theory) below).

To benchmark the trained autoencoder, we use it to obtain embeddings for images on the train and test set, and then train a multiclass logistic regression on top of these features, and measure its accuracy.

## Training
    
    $ git clone http://github.com/simaoh/autoencoders_benchmark
    $ cd autoencoders_benchmark
    $ python autoencoder.py --logdir output_file

## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard.
    
    $ tensorboard --logdir=output_file

You will be able to inspect the underlying graph of the autoencoder, getting a better understanding of how: 
- respective encoding and decoding layers are chosen to share the same convolution filters.
- noise and dropout help in better learning image features, by difficulting the reconstruction task. 
- batch normalization helps in faster training time.

## Theory

Autoencoders are neural networks trained to reconstruct their input.
For this to be a non-trivial task, several restrictions are imposed on the NN architecture. Most tipically:
 - the hidden layers are required to have a much smaller number of units.
 - noise is added to the input to be reconstructed
 - the derivative of the encoding function is bounded
This prevents the neural network from simply memorizing the image, forcing it to learn its main features. 

For an introduction to autoencoders,  [this introductory blog post explaining how it generalizes PCA](https://lazyprogrammer.me/a-tutorial-on-autoencoders/), and Goodfell-et-all-2016 [Deep learning book-Chapter 14](http://www.deeplearningbook.org/contents/autoencoders.html)
are excellent sources.

Our Denoising Autoencoder looks like this
<img src=https://github.com/simaoh/autoencoders_benchmark/blob/master/autoencoder_graph.png?raw=true></img>
