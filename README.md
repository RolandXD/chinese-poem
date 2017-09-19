
* 直接运行：
``` python
 python  poem_char_rnn.py
``` 

* 运行日志
``` 
PS D:\learning\gi_workspace\lishuai-github\中文诗词生成> & D:/learning/anaconda3-x64/envs/py35-x64/python d:/learning/gi_workspa
ce/lishuai-github/中文诗词生成/poem_char_rnn.py
2017-09-19 13:26:09,307 : INFO : load dataset  from ./data/dataset.pkl
2017-09-19 13:26:09,572 : DEBUG : poem number is 42975
2017-09-19 13:26:09,572 : DEBUG : dict size is 7511
2017-09-19 13:26:09,572 : DEBUG : one segment :宵合，琼楼拂曙通。年光三月里，宫殿百花中
2017-09-19 13:26:09.594321: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guar
d.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed
 up CPU computations.
2017-09-19 13:26:09.594430: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guar
d.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could spee
d up CPU computations.
2017-09-19 13:26:10.640077: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_de
vice.cc:955] Found device 0 with properties:
name: GeForce GTX 850M
major: 5 minor: 0 memoryClockRate (GHz) 0.9015
pciBusID 0000:0a:00.0
Total memory: 4.00GiB
Free memory: 3.35GiB
2017-09-19 13:26:10.640192: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_de
vice.cc:976] DMA: 0
2017-09-19 13:26:10.640691: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_de
vice.cc:986] 0:   Y
2017-09-19 13:26:10.640846: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_de
vice.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 850M, pci bus id: 0000:0a:00.0)
2017-09-19 13:26:10,642 : DEBUG : batch_size is 512,num_steps is 32
  [TL] EmbeddingInputlayer model/embeding: (7511, 128)
  [TL] RNNLayer model/encode_lstm: n_hidden:200 n_steps:32 in_dim:3 in_shape:(512, 32, 128) cell_fn:BasicLSTMCell
       RNN batch_size (concurrent processes): 512
     n_params : 2
  [TL] RNNLayer model/decode_lstm: n_hidden:200 n_steps:32 in_dim:3 in_shape:(512, 32, 200) cell_fn:BasicLSTMCell
       RNN batch_size (concurrent processes): 512
     n_params : 2
  [TL] DenseLayer  model/output: 7511 identity
  [TL] EmbeddingInputlayer model/embeding: (7511, 128)
  [TL] RNNLayer model/encode_lstm: n_hidden:200 n_steps:1 in_dim:3 in_shape:(1, 1, 128) cell_fn:BasicLSTMCell
       RNN batch_size (concurrent processes): 1
     n_params : 2
  [TL] RNNLayer model/decode_lstm: n_hidden:200 n_steps:1 in_dim:3 in_shape:(1, 1, 200) cell_fn:BasicLSTMCell
       RNN batch_size (concurrent processes): 1
     n_params : 2
  [TL] DenseLayer  model/output: 7511 identity
  param   0: model/embeding/embeddings:0 (7511, 128)        float32_ref (mean: 4.8544803576078266e-05, median: 7.843971252441406
e-05, std: 0.05773523077368736)
  param   1: model/encode_lstm/basic_lstm_cell/kernel:0 (328, 800)         float32_ref (mean: -0.00013741284783463925, median: -
0.00024099647998809814, std: 0.05768219754099846)
  param   2: model/encode_lstm/basic_lstm_cell/bias:0 (800,)             float32_ref (mean: 0.0               , median: 0.0
          , std: 0.0               )
  param   3: model/decode_lstm/basic_lstm_cell/kernel:0 (400, 800)         float32_ref (mean: 1.1575039934541564e-05, median: -0
.00013854354619979858, std: 0.05774073675274849)
  param   4: model/decode_lstm/basic_lstm_cell/bias:0 (800,)             float32_ref (mean: 0.0               , median: 0.0
          , std: 0.0               )
  param   5: model/output/W:0     (200, 7511)        float32_ref (mean: 7.735443068668246e-05, median: 0.00012183739454485476, s
td: 0.0879872590303421)
  param   6: model/output/b:0     (7511,)            float32_ref (mean: 0.0               , median: 0.0               , std: 0.0
               )
  num of params: 3055119
  layer   0: model/embeding/embedding_lookup:0 (512, 32, 128)     float32
  layer   1: model/Reshape:0      (512, 32, 200)     float32
  layer   2: model/Reshape_1:0    (16384, 200)       float32
  layer   3: model/output/Identity:0 (16384, 7511)      float32
  [*] printing global variables
  var   0: (7511, 128)       model/embeding/embeddings:0
  var   1: (328, 800)        model/encode_lstm/basic_lstm_cell/kernel:0
  var   2: (800,)            model/encode_lstm/basic_lstm_cell/bias:0
  var   3: (400, 800)        model/decode_lstm/basic_lstm_cell/kernel:0
  var   4: (800,)            model/decode_lstm/basic_lstm_cell/bias:0
  var   5: (200, 7511)       model/output/W:0
  var   6: (7511,)           model/output/b:0
  var   7: ()                beta1_power:0
  var   8: ()                beta2_power:0
  var   9: (7511, 128)       model/embeding/embeddings/Adam:0
  var  10: (7511, 128)       model/embeding/embeddings/Adam_1:0
  var  11: (328, 800)        model/encode_lstm/basic_lstm_cell/kernel/Adam:0
  var  12: (328, 800)        model/encode_lstm/basic_lstm_cell/kernel/Adam_1:0
  var  13: (800,)            model/encode_lstm/basic_lstm_cell/bias/Adam:0
  var  14: (800,)            model/encode_lstm/basic_lstm_cell/bias/Adam_1:0
  var  15: (400, 800)        model/decode_lstm/basic_lstm_cell/kernel/Adam:0
  var  16: (400, 800)        model/decode_lstm/basic_lstm_cell/kernel/Adam_1:0
  var  17: (800,)            model/decode_lstm/basic_lstm_cell/bias/Adam:0
  var  18: (800,)            model/decode_lstm/basic_lstm_cell/bias/Adam_1:0
  var  19: (200, 7511)       model/output/W/Adam:0
  var  20: (200, 7511)       model/output/W/Adam_1:0
  var  21: (7511,)           model/output/b/Adam:0
  var  22: (7511,)           model/output/b/Adam_1:0
2017-09-19 13:26:30,045 : INFO : load model param from model.npz
2017-09-19 13:26:32,183 : DEBUG : 0-0  perplexity: 56.23729316474241
2017-09-19 13:27:39,017 : DEBUG : 0-100  perplexity: 48.175525867798214
2017-09-19 13:28:37,504 : DEBUG : Epoch: 1/1000 Train Perplexity: 49.235688249977855
[*] model.npz saved
2017-09-19 13:28:38,547 : INFO : new poem : 又产三春独足劳，无凭风雨绕林园。当时更想招贤事，坐忆明光又九秋。
2017-09-19 13:28:39,428 : DEBUG : 1-0  perplexity: 49.251626861679455
2017-09-19 13:29:46,185 : DEBUG : 1-100  perplexity: 48.715132005984756
2017-09-19 13:30:44,345 : DEBUG : Epoch: 2/1000 Train Perplexity: 49.21514053651277
[*] model.npz saved
2017-09-19 13:30:44,772 : INFO : new poem : 铸金六十八千回，累里烟花一度春。坐使欲归非旧灶，卷帷春日是嘉年。
2017-09-19 13:30:45,621 : DEBUG : 2-0  perplexity: 49.2273593163576

```

