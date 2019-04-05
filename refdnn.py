import os
import math
import numpy as np
import time
import tensorflow as tf

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def logging_time(original_function):
    def wrapper_function(*args, **kwargs):
        start_time = time.time()
        result = original_function(*args, **kwargs)
        print('-------- {}: {:.3f} sec -------'.format(original_function.__name__, (time.time()-start_time)))
        return result
    return wrapper_function
    
class REFDNN:
    def __init__(self,
                 hidden_units,
                 learning_rate_ftrl,
                 learning_rate_adam,
                 l1_regularization_strength,
                 l2_regularization_strength,
                 batch_size=32,
                 training_steps=10000,
                 evaluation_steps=100,
                 earlystop_use=True, patience=20,
                 gpu_use=False,
                 checkpoint_path=None):
        ## hyperparameters
        self.hidden_units = hidden_units
        self.learning_rate_ftrl = learning_rate_ftrl
        self.learning_rate_adam = learning_rate_adam
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        ## parameters
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps
        self.earlystop_use = earlystop_use
        self.patience = patience
        ## environment
        self.gpu_use = gpu_use
        if checkpoint_path is None:
            checkpoint_path = "ckpt_RefDNN.ckpt"
        self.checkpoint_path = checkpoint_path
    

    def fit(self,
            X_train, S_train, I_train, Y_train,
            X_valid, S_valid, I_valid, Y_valid,
            verbose=2):
        
        ## constant
        epsilon = 1e-5
        threshold = 0.5
        
        ## data information
        self._X_shape = (None, X_train.shape[1]) # number of genes
        self._S_shape = (None, S_train.shape[1]) # number of drugs
        
        ## tf.Session
        self._open_session()
        
        ## tf.placeholder
        X_train_PH = tf.placeholder(shape=self._X_shape, dtype=tf.float32, name='X_train_PH')
        S_train_PH = tf.placeholder(shape=self._S_shape, dtype=tf.float32, name='S_train_PH')
        I_train_PH = tf.placeholder(shape=(None,), dtype=tf.int32, name='I_train_PH')
        Y_train_PH = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_train_PH')
        
        X_valid_PH = tf.placeholder(shape=self._X_shape, dtype=tf.float32, name='X_valid_PH')
        S_valid_PH = tf.placeholder(shape=self._S_shape, dtype=tf.float32, name='S_valid_PH')
        I_valid_PH = tf.placeholder(shape=(None,), dtype=tf.int32, name='I_valid_PH')
        Y_valid_PH = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_valid_PH')
        
        ## tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((X_train_PH, S_train_PH, I_train_PH, Y_train_PH))
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.shuffle(buffer_size=10000, seed=2019)
        dataset_train = dataset_train.batch(self.batch_size, drop_remainder=False)
        dataset_train.prefetch(2 * self.batch_size)
        
        dataset_valid = tf.data.Dataset.from_tensor_slices((X_valid_PH, S_valid_PH, I_valid_PH, Y_valid_PH))
        dataset_valid = dataset_valid.repeat()
        dataset_valid = dataset_valid.batch(self.batch_size, drop_remainder=False)
        dataset_valid.prefetch(2 * self.batch_size)
        
        ## tf.iterator
        iterator_train = dataset_train.make_initializable_iterator()
        iterator_valid = dataset_valid.make_initializable_iterator()
        _ = self.sess.run(iterator_train.initializer,
                          feed_dict={X_train_PH:X_train,
                                     S_train_PH:S_train,
                                     I_train_PH:I_train,
                                     Y_train_PH:Y_train})
        _ = self.sess.run(iterator_valid.initializer,
                          feed_dict={X_valid_PH:X_valid,
                                     S_valid_PH:S_valid,
                                     I_valid_PH:I_valid,
                                     Y_valid_PH:Y_valid})
                                     
        ## tf.iterator.handle
        handle_train = self.sess.run(iterator_train.string_handle())
        handle_valid = self.sess.run(iterator_valid.string_handle())
        
        ## tf.graph
        self._output_types = dataset_train.output_types
        self._output_shapes = dataset_train.output_shapes
        
        self._create_graph()
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        
        ## history
        history = {
            'train':{
                'loss':[]
            },
            'valid':{
                'loss':[],
                'acc':[],
                'precision':[],
                'recall':[]
            }
        }
        
        ## initialization for early stopping
        if self.earlystop_use:
            best_loss = np.Inf
            best_step = 0
            cnt_patience = 0
            
        ## initialize terminal conditions
        termination_convergence = False
        termination_earlystop = False
        
        ## start fitting
        start_time = time.time()
        for step in range(1, self.training_steps+1):
            ## 1) training 
            _, _, loss_train = self.sess.run([self.TRAIN_ftrl, self.TRAIN_adam, self.LOSS],
                                             feed_dict={self.handle:handle_train, self.training:True})
                        
            ## 2) evaluation
            if step % self.evaluation_steps == 0:
                ## 2-2) validation loss
                loss_valid = 0.
                acc_valid = 0.
                precision_valid = 0.
                recall_valid = 0.
                n_batch_valid = math.ceil(len(X_valid) // self.batch_size)
                for _ in range(n_batch_valid):
                    metrics_valid = self.sess.run([self.LOSS, self.ACCURACY, self.PRECISION, self.RECALL],
                                                  feed_dict={self.handle:handle_valid, self.training:False})
                    loss_valid += metrics_valid[0]
                    acc_valid += metrics_valid[1][0]
                    precision_valid += metrics_valid[2][0]
                    recall_valid += metrics_valid[3][0]
                loss_valid /= n_batch_valid
                acc_valid /= n_batch_valid
                precision_valid /= n_batch_valid
                recall_valid /= n_batch_valid
                
                ## 2-3) store
                history['train']['loss'].append((step, loss_train))
                history['valid']['loss'].append((step, loss_valid))
                history['valid']['acc'].append((step, acc_valid))
                history['valid']['precision'].append((step, precision_valid))
                history['valid']['recall'].append((step, recall_valid))
                
                if verbose > 1:
                    end_time = time.time()
                    log = "[RefDNN][{:05d}] LOSS_train={:.5f} | LOSS_valid={:.5f}".format(step, loss_train, loss_valid)
                    log += " | ACC_valid={:.3f}".format(acc_valid)
                    log += " | PRECISION_valid={:.3f}".format(precision_valid)
                    log += " | RECALL_valid={:.3f}".format(recall_valid)
                    log += " | ({:.3f} sec)".format(end_time-start_time)
                    print(log)
                    start_time = time.time()
                    
                ## 2-4) early stopping
                if self.earlystop_use:
                    if loss_valid < best_loss:
                        best_loss = loss_valid
                        best_step = step
                        cnt_patience = 0
                        self.saver.save(self.sess, self.checkpoint_path)
                        if verbose > 1:
                            print("[RefDNN][CHECKPOINT] Model is saved in: {}".format(self.checkpoint_path))
                    elif cnt_patience < self.patience:
                        cnt_patience += 1
                    else:
                        termination_earlystop = True
                        
                else:
                    self.saver.save(self.sess, self.checkpoint_path)
                    if verbose > 1:
                        print("[RefDNN][CHECKPOINT] Model is saved in: {}".format(self.checkpoint_path))

            ## 3) convergence
            if len(history['train']['loss']) > 1:
                termination_convergence = abs(history['train']['loss'][-1][1] - history['train']['loss'][-2][1]) < epsilon
            
            ## 4) termination
            if termination_convergence or termination_earlystop:
                break
                
        ## tf.Session
        self._close_session()
        return history
    

    def predict(self, X_test, S_test, verbose=2):
        ## dummy input data
        I_test = np.zeros((X_test.shape[0],), dtype=np.int32)
        Y_test = np.zeros((X_test.shape[0],1), dtype=np.uint8)
        
        ## tf.Session
        self._open_session()
        
        ## tf.placeholder
        X_test_PH = tf.placeholder(shape=self._X_shape, dtype=tf.float32, name='X_test_PH')
        S_test_PH = tf.placeholder(shape=self._S_shape, dtype=tf.float32, name='S_test_PH')
        I_test_PH = tf.placeholder(shape=(None,), dtype=tf.int32, name='I_test_PH')
        Y_test_PH = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_test_PH')
        
        ## tf.dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test_PH, S_test_PH, I_test_PH, Y_test_PH))
        dataset_test = dataset_test.batch(self.batch_size)
        dataset_test.prefetch(2 * self.batch_size)
        
        ## tf.iterator
        iterator_test = dataset_test.make_initializable_iterator()
        _ = self.sess.run(iterator_test.initializer,
                          feed_dict={X_test_PH:X_test,
                                     S_test_PH:S_test,
                                     I_test_PH:I_test,
                                     Y_test_PH:Y_test})
        ## tf.iterator.handle
        handle_test = self.sess.run(iterator_test.string_handle())
        
        ## tf.graph
        self._create_graph()
        self.saver.restore(self.sess, self.checkpoint_path)
        if verbose > 1:
            print("[RefDNN][CHECKPOINT] Model is restored from: {}".format(self.checkpoint_path))
        
        ## initialization of outputs
        outputs = []
        
        ## predict per batch
        while True:
            try:
                predictions = self.sess.run(self.Y_main, feed_dict={self.handle:handle_test, self.training:False})
                outputs += predictions.tolist()
            except tf.errors.OutOfRangeError:
                break
                
        ## tf.Session
        self._close_session()
        return np.array(outputs, dtype=np.uint8)
    

    def predict_proba(self, X_test, S_test, verbose=2):
        ## dummy input data
        I_test = np.zeros((X_test.shape[0],), dtype=np.int32)
        Y_test = np.zeros((X_test.shape[0],1), dtype=np.uint8)
        
        ## tf.Session
        self._open_session()
        
        ## tf.placeholder
        X_test_PH = tf.placeholder(shape=self._X_shape, dtype=tf.float32, name='X_test_PH')
        S_test_PH = tf.placeholder(shape=self._S_shape, dtype=tf.float32, name='S_test_PH')
        I_test_PH = tf.placeholder(shape=(None,), dtype=tf.int32, name='I_test_PH')
        Y_test_PH = tf.placeholder(shape=(None,1), dtype=tf.uint8, name='Y_test_PH')
        
        ## tf.dataset
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test_PH, S_test_PH, I_test_PH, Y_test_PH))
        dataset_test = dataset_test.batch(self.batch_size)
        dataset_test.prefetch(2 * self.batch_size)
        
        ## tf.iterator
        iterator_test = dataset_test.make_initializable_iterator()
        _ = self.sess.run(iterator_test.initializer,
                          feed_dict={X_test_PH:X_test,
                                     S_test_PH:S_test,
                                     I_test_PH:I_test,
                                     Y_test_PH:Y_test})
        ## tf.iterator.handle
        handle_test = self.sess.run(iterator_test.string_handle())
        
        ## tf.graph
        self._create_graph()
        self.saver.restore(self.sess, self.checkpoint_path)
        if verbose > 1:
            print("[RefDNN][CHECKPOINT] Model is restored from: {}".format(self.checkpoint_path))
        
        ## initialization of outputs
        outputs = []
        
        ## predict per batch
        while True:
            try:
                probabilites = self.sess.run(self.P_main, feed_dict={self.handle:handle_test, self.training:False})
                outputs += probabilites.tolist()
            except tf.errors.OutOfRangeError:
                break
                
        ## tf.Session
        self._close_session()
        return np.array(outputs, dtype=np.float32)
    
    
    def get_kernels(self, hidden_names=None, verbose=2):
        '''
        dense0
        dense1
        dense2
        output
        '''
        if hidden_names == None:
            hidden_names = ['dense0', 'dense1', 'dense2', 'output']
            
        ## tf.Session
        self._open_session()
        ## tf.graph
        self._create_graph()
        self.saver.restore(self.sess, self.checkpoint_path)
        if verbose > 1:
            print("[RefDNN][CHECKPOINT] Model is restored from: {}".format(self.checkpoint_path))
        ## weights
        kernel_dict = {}
        for hidden_name in hidden_names:
            weights = tf.get_default_graph().get_tensor_by_name('{}/kernel:0'.format(hidden_name))
            kernel_dict[hidden_name] = self.sess.run(weights)
        ## tf.Session
        self._close_session()
        return kernel_dict
        
        
    def _create_graph(self):
        ## tf.Placeholder
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training = tf.placeholder(dtype=bool, name='training_PH')
        
        ## tf.Iterator
        iterator = tf.data.Iterator.from_string_handle(string_handle=self.handle,
                                                       output_types=self._output_types,
                                                       output_shapes=self._output_shapes)
        X_batch, S_batch, I_batch, Y_batch = iterator.get_next()

        ## Activation function
        nonlinear = tf.nn.sigmoid
        
        ## Model
        dense0    = tf.layers.dense(inputs=X_batch, units=self._S_shape[1], activation='linear', name='dense0')
        activate0 = nonlinear(dense0, name='activation0')
        
        dense1    = tf.layers.dense(inputs=tf.multiply(S_batch, activate0), units=self.hidden_units, activation='linear', name='dense1')
        bn1       = tf.layers.batch_normalization(dense1, training=self.training, name='bn1')
        activate1 = nonlinear(bn1, name='activation1')
        
        dense2    = tf.layers.dense(inputs=activate1, units=self.hidden_units, activation='linear', name='dense2')
        bn2       = tf.layers.batch_normalization(dense2, training=self.training, name='bn2')
        activate2 = nonlinear(bn2, name='activation2')
        
        ## Output
        O_ELASTICNET  = tf.expand_dims(tf.reduce_sum(tf.multiply(dense0, tf.one_hot(I_batch, depth=self._S_shape[1])), 1), axis=-1)
        O_DNN = tf.layers.dense(inputs=activate2, units=1, activation='linear', name='output')
        
        ## LOSS
        LOSS_ELASTICNET  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=O_ELASTICNET, labels=tf.cast(Y_batch, dtype=tf.float32)))
        LOSS_DNN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=O_DNN, labels=tf.cast(Y_batch, dtype=tf.float32)))
        self.LOSS = 0.5 * (LOSS_DNN + LOSS_ELASTICNET)
        
        ## OPTIMIZER 1st
        OPT_ftrl = tf.train.FtrlOptimizer(learning_rate=self.learning_rate_ftrl,
                                           l1_regularization_strength=self.l1_regularization_strength,
                                           l2_regularization_strength=self.l2_regularization_strength)
        var_list_ftrl = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense0')
        
        ## OPTIMIZER 2nd
        OPT_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate_adam)
        var_list_adam = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense1')
        var_list_adam += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense2')
        var_list_adam += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'output')
        var_list_adam += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'batchnormalization1')
        var_list_adam += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'batchnormalization2')
        
        ## MINIMIZATION
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.TRAIN_ftrl = OPT_ftrl.minimize(self.LOSS, var_list=var_list_ftrl)
            self.TRAIN_adam = OPT_adam.minimize(self.LOSS, var_list=var_list_adam)
        
        ## PREDICTION
        self.P_main = tf.sigmoid(O_DNN)
        self.Y_main = tf.cast(self.P_main > 0.5, tf.uint8)
        
        ## ACCURACY
        self.ACCURACY = tf.metrics.accuracy(labels=Y_batch, predictions=self.Y_main)
        ## PRECISION
        self.PRECISION = tf.metrics.precision(labels=Y_batch, predictions=self.Y_main)
        ## RECALL
        self.RECALL = tf.metrics.recall(labels=Y_batch, predictions=self.Y_main)
        
        ## SAVE AND RESTORE
        self.saver = tf.train.Saver()
        
    def _open_session(self):
        ## Create a new session
        tf.reset_default_graph()
        if self.gpu_use:
            ## GPU
            gpu_options = tf.GPUOptions()
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            ## CPU
            self.sess = tf.Session()
        
    def _close_session(self):
        self.sess.close()