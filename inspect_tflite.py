import tensorflow as tf
p='app/src/main/assets/gpt2/gpt2-64-8bits.tflite'
i=tf.lite.Interpreter(model_path=p); i.allocate_tensors()
for group, items in [('inputs',i.get_input_details()),('outputs',i.get_output_details())]:
 print(group)
 for x in items:
  print(x['index'], x['name'], x['shape'].tolist(), x['dtype'].__name__, 'quant', x['quantization'])
