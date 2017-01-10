import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from model import autoencoder,decode, variable_summaries
import argparse



## Data- MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets(train_dir="/tmp/data/",one_hot=True)


def train(args):
  
    ##parameters
    n_steps=args.n_steps
    n_input=args.n_input
    batch_size=args.batch_size
    dropout_constant=args.dropout_constant
    
    
    ## Underlying graph
    graph = tf.Graph()
    with graph.as_default():

        #data
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, n_input], name='x-input')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)

        with tf.name_scope('noise'):
            noise_std = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', noise_std)
        



        #model
        code, reconstruction=autoencoder(x,dropout=keep_prob, noise_std=noise_std)

        #loss
        with tf.name_scope('MSE_loss'):
            loss=tf.reduce_sum(tf.square(x - reconstruction))
            global_step = tf.Variable(0)  # count the number of steps taken.
            learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.91)
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
            tf.summary.scalar("loss",loss)


        #logging
        log_dir=args.log_dir
        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train',graph=graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')



        #training  
        sess = tf.InteractiveSession(graph=graph)
        tf.global_variables_initializer().run()

        mean_img_train = np.mean(mnist.train.images[:50000], axis=0)
        mean_img_validation= np.mean(mnist.train.images[50000:55000], axis=0)
        x_valid=[img-mean_img_validation for img in mnist.train.images][50000:55000]
        
        for i in range(n_steps):       
            if i %10==0:
                #test error
                cost,summary=sess.run([loss,merged], feed_dict={x:x_valid,keep_prob:1.0, noise_std:0})
                test_writer.add_summary(summary, i)
                print "Loss at step %d: %.1f" %(i,cost)
            else:  
                # Record train set summaries and train
                offset = (i * batch_size) % (50000 - batch_size)
                # Generate a minibatch.
                batch_data = mnist.train.images[offset:(offset + batch_size), :]
                x_batch= np.array([img - mean_img_train for img in batch_data])

                # Record execution stats (every 100th step)
                if i % 100 == 99:  

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _,summary= sess.run([optimizer,merged],
                                          feed_dict={x:x_batch, keep_prob:dropout_constant, noise_std:0.05},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)

                else:   
                    _,cost,summary=sess.run([optimizer, loss,merged], 
                                            feed_dict={x:x_batch, keep_prob:dropout_constant, noise_std:0.05})
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()
        print ("For Autoencoder tensorboard logs run 'tensorboard --logdir=%s'" %log_dir)
        
        
        #Autoencoder Benchmarking
        
        #Embeddings from encoded train set
        train_size=55000
        mean_img_train = np.mean(mnist.train.images[:train_size], axis=0)

        for i in range(train_size/100):
            step=i*100
            x_mnist_train, y_mnist_train=mnist.train.images[step:step+100],mnist.train.labels[step:step+100]
            x_mnist_train_norm = np.array([img - mean_img_train for img in x_mnist_train])
            new_embedds=sess.run(code, feed_dict={x: x_mnist_train_norm, keep_prob:1, noise_std:0})
            if i ==0:
                train_embedds=new_embedds
            else:
                train_embedds=np.concatenate((train_embedds,new_embedds),axis=0)
        
        #Multiclass LogRegression
        clf=LogisticRegression(multi_class='multinomial',solver='lbfgs')
        clf.fit(train_embedds, np.argmax(mnist.train.labels[:train_size],axis=1))
        
        #Embeddings from encoded train set
        test_size=10000
        mean_img_test = np.mean(mnist.test.images, axis=0)
        for i in range(test_size/100):
            step=i*100
            x_mnist_test, y_test=mnist.test.images[step:step+100],mnist.test.labels[step:step+100]
            x_mnist_test_norm = np.array([img - mean_img_train for img in x_mnist_test])
            new_embedds=sess.run(code, feed_dict={x: x_mnist_test_norm,keep_prob:1, noise_std:0})
            if i ==0:
                test_embedds=new_embedds
            else:
                test_embedds=np.concatenate((test_embedds,new_embedds),axis=0)
        
        #LogRegression Accuracy
        accuracy=np.mean(clf.predict(test_embedds)==np.argmax(mnist.test.labels[:test_size],axis=1))
        print('Accuracy: %.2f%%' %(accuracy*100))
        
        
        

        
def parse_args():
    
    desc = "TensorFlow implementation of 'Denoising Autoencoder'"  
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--n_steps', type=int, default=200,
                        help='number of batches to train over')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of each minibatch')
    parser.add_argument('--n_input', type=int, default=784,
                        help='image side lenght')
    parser.add_argument('--dropout_constant', type=float, default=0.8,
                        help='fraction of nodes kept in the dropout layer')
    parser.add_argument('--log_dir', type=str, default='/tmp/auto/',
                        help='directory where to save tensorboard logs')
    
    args = parser.parse_args()
    return args
          

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()