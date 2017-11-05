# coding=utf-8
''' poemCharRNN 模型
 -keras 简单，不灵活。
 -tensorlayer，既简单，又灵活。[使用]
'''


import os  
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import logging
import pickle
import numpy as np
import glob
import tensorflow  as tf 
import tensorlayer as tl

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
BOS = '<s>'
EOS = '</s>'
UNK = '<unk>'

#列出目录内文件,不做递归
def dirfileWithRecur( path , files ):
    if os.path.isfile(path):
        files.append(path)
        return files

    for fn in glob.glob( path + os.sep + '*' ): # '*'代表匹配所有文件
        if os.path.isdir( fn ):
            dirfileWithRecur( fn, files)   #遍历目录内文件
            continue   
        else:
            files.append(fn)
    return files

class PoemCharRNN:
    def load_language(self,dir, store_file="./data/pythondataset.pkl"):
        '''加载源代码文件，'''
        if  os.path.isfile(store_file):
            with open(store_file, 'rb') as f:
                logging.info('load dataset  from {}'.format(store_file))
                self.word_to_id, self.id_to_word, data = pickle.load(f)
                return data

        files=[]
        dirfileWithRecur(dir,files)

        data=[]
        dic = set()
        dic.add(BOS)
        dic.add(EOS)
        dic.add(UNK)
        for fname in files:
            if not  fname.endswith('.py'):
                continue
            with open(fname,encoding='utf-8') as f:
                doc = f.read()
            data.append(BOS)
            data.extend(list(doc))
            data.append(EOS)
            dic=dic.union(set(list(doc)))
        dic = list(dic)
        self.word_to_id = {dic[i]: i for i in range(len(dic))}
        self.id_to_word = {i: dic[i] for i in range(len(dic))}
   
        data =[ self.word_to_id[w] for w in data ]
        with open(store_file, 'wb') as f:
            pickle.dump((self.word_to_id, self.id_to_word, data), f)

        logging.debug('doc number is {} '.format(data.count(self.word_to_id[EOS])))
        logging.debug('dict size is {} '.format(len(self.word_to_id)))
        return data


    def load_data(self, fname, store_file="./data/dataset.pkl",poem_length=None):
        '''
            加载数据
            数据格式：
                903_21	同前（崔十娘）	张鷟	映水俱知笑，成蹊竟不言。即今无自在，高下任渠攀 
                901_17	送别（含思落句势）	王昌龄	春江愁送君，蕙草生氤氲。醉后不能语，乡山雨雰雰。

        '''
        if  False : #os.path.isfile(store_file):
            with open(store_file, 'rb') as f:
                logging.info('load dataset  from {}'.format(store_file))
                self.word_to_id, self.id_to_word, data = pickle.load(f)
                return data
        
        logging.info('load raw data   from {}'.format(fname))

        data = []
        dic = [UNK]
        with open(fname, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                segs = line.strip().split('\t')
                if len(segs) < 4:
                    logging.warn('read line error :{}'.format(line))
                    continue
                text = [BOS]
                text.extend(list(segs[3]))
                text.append(EOS)
                if poem_length != None and len(text)!= poem_length+2:
                    continue 
                data.extend(text)
        dic.extend(list(set(data)))
        
        self.word_to_id = {dic[i]: i for i in range(len(dic))}
        self.id_to_word = {i: dic[i] for i in range(len(dic))}
           
        data =[ self.word_to_id[w] for w in data ]
        with open(store_file, 'wb') as f:
            pickle.dump((self.word_to_id, self.id_to_word, data), f)

        logging.debug('poem number is {} '.format(data.count(self.word_to_id[EOS])))
        logging.debug('dict size is {} '.format(len(self.word_to_id)))
        return data


    def to_word(self,predict, vocabs):
        t = np.cumsum(predict)
        s = np.sum(predict)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        if sample > len(vocabs):
            sample = len(vocabs) - 1
        return vocabs[sample]


    def train_and_infer(self, data, epoch=20, model_file='model.npz',infer_length=64):
        '''训练模型'''

        sess =tf.InteractiveSession()
        #设定参数
        batch_size = 512
        num_steps  = 32
        vocab_size = len(self.word_to_id) 
        embed_size=128
        hidden_size = 200
        logging.debug('batch_size is {},num_steps is {}'.format(batch_size,num_steps))

        # 构建模型
        input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        input_data_infer = tf.placeholder(tf.int32, [1, 1])
        targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        def inference(x,is_training,num_steps,batch_size,reuse=None):
            with tf.variable_scope("model", reuse=reuse):
                tl.layers.set_name_reuse(reuse)
                network=tl.layers.EmbeddingInputlayer(inputs=x,
                                                    vocabulary_size = vocab_size,
                                                    embedding_size=embed_size,
                                                    E_init = tf.random_uniform_initializer(-0.1, 0.1),
                                                    name='embeding')
                network = tl.layers.RNNLayer(network,
                                            cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                            cell_init_args={'state_is_tuple': True},
                                            return_last=False,
                                            n_steps = num_steps,
                                            n_hidden = hidden_size,
                                            name='encode_lstm')
                
                encode_lstm = network  
                network = tl.layers.RNNLayer(network,
                                            cell_fn=tf.contrib.rnn.BasicLSTMCell,
                                            cell_init_args = {'state_is_tuple':True},
                                            return_last = False,
                                            n_steps = num_steps,
                                            n_hidden = hidden_size,
                                            return_seq_2d=True,
                                            name='decode_lstm')
                decode_lstm =network   ##### 这不就是 双层 rnn吗？
                network = tl.layers.DenseLayer(network,n_units=vocab_size,name='output')
            return network,encode_lstm,decode_lstm

        network,encode_lstm,decode_lstm =inference(input_data,is_training=True,num_steps=num_steps,batch_size=batch_size)
        network_infer,encode_lstm_infer,decode_lstm_infer =inference(input_data_infer,is_training=False,num_steps=1,batch_size=1,reuse=True)
        infer_out =tf.nn.softmax( network_infer.outputs)
        def loss_fn(output,targets):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [output],
                [tf.reshape(targets,[-1])],
                [tf.ones_like(tf.reshape(targets, [-1]),dtype=tf.float32)])
            cost =tf.reduce_sum(loss)/batch_size
            return cost

        cost = loss_fn(network.outputs,targets)

        #优化cost
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        #初始化变量
        tl.layers.initialize_global_variables(sess)
        network.print_params()
        network.print_layers()
        tl.layers.print_all_variables()

        #加载保存的模型
        if os.path.isfile(model_file):
            logging.info('load model param from {}'.format(model_file))
            param  =tl.files.load_npz(path='',name=model_file)
            tl.files.assign_params(sess,param,network)
        else :
            logging.info('start learning  from beginning ... ...')

        costs = 0.0; iters = 0
        epoch_size = ((len(data) // batch_size) - 1) // num_steps
        for i in range(epoch):
            encode_state = tl.layers.initialize_rnn_state(encode_lstm.initial_state)
            decode_state = tl.layers.initialize_rnn_state(decode_lstm.initial_state)

            for step,(x,y) in enumerate(tl.iterate.ptb_iterator(data,batch_size=batch_size,num_steps=num_steps)):
                feed_dict = {input_data:x,targets:y,
                            encode_lstm.initial_state :encode_state,
                            decode_lstm.initial_state : decode_state}
                            
                _cost,encode_state,decode_state,_ =sess.run([cost,encode_lstm.final_state,decode_lstm.final_state,train_op],feed_dict=feed_dict)
                costs+=_cost 
                iters+=num_steps

                if step % 100 == 0:
                    logging.debug("{}-{}  perplexity: {}".format(i,step,np.exp(costs / iters)))
            logging.debug("Epoch: {}/{} Train Perplexity: {}".format(i + 1, epoch,np.exp(costs / iters)))
            tl.files.save_npz(network.all_params , name=model_file)

            #任意作诗一首
            encode_state = tl.layers.initialize_rnn_state(encode_lstm_infer.initial_state)
            decode_state = tl.layers.initialize_rnn_state(decode_lstm_infer.initial_state)
            poem = [BOS]
            for w in poem:
                tx =  np.array([[self.word_to_id[w]]])
                feed_dict = {input_data_infer:tx,
                            encode_lstm_infer.initial_state :encode_state,
                            decode_lstm_infer.initial_state : decode_state}
                out,encode_state,decode_state = sess.run([infer_out,encode_lstm_infer.final_state,decode_lstm_infer.final_state],feed_dict=feed_dict)

            word = self.to_word(out,self.id_to_word )
            while word != EOS  and len(poem) < infer_length :
                poem.append(word)
                feed_dict = {input_data_infer:[[self.word_to_id[word]]],
                             encode_lstm_infer.initial_state :encode_state,
                            decode_lstm_infer.initial_state : decode_state}
                out,encode_state,decode_state = sess.run([infer_out,encode_lstm_infer.final_state,decode_lstm_infer.final_state],feed_dict=feed_dict)
                word = self.to_word(out,self.id_to_word ) #概率选择保证了每次做的诗时不一样的
            logging.info('new text : \n#################\n{}\n#################\n'.format(''.join(poem[1:])))
           

if __name__ == '__main__':
    pcrnn = PoemCharRNN()

    #只用四言绝句训练
    #data = pcrnn.load_data('./data/ts.txt',poem_length=24)
    #pcrnn.train_and_infer(data,epoch=1000,model_file='model-s4-e1000.npz')

    data =pcrnn.load_language("D:\\learning\\data\\synitalent\\代码数据集\\gitprojects\\Python@boto")
    pcrnn.train_and_infer(data,epoch=1000,model_file='model-python-e1000.npz',infer_length=1024)

    #data = pcrnn.load_data('./data/ts.txt')
    #pcrnn.train_and_infer(data)
    
  
    
