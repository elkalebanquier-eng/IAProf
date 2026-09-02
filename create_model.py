import tensorflow as tf
import numpy as np
from pathlib import Path

class TinyTeacher(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 64], tf.int32, name='token_ids')])
    def serving_default(self, token_ids):
        # Constant, compact neural signal: token IDs -> four local logits.
        emb = tf.constant(np.random.default_rng(7).normal(0, 0.1, (256, 8)).astype(np.float32))
        weights = tf.constant(np.random.default_rng(8).normal(0, 0.1, (8, 4)).astype(np.float32))
        bias = tf.constant(np.zeros(4, dtype=np.float32))
        pooled = tf.reduce_mean(tf.gather(emb, token_ids), axis=1)
        return {'logits': tf.linalg.matmul(pooled, weights) + bias}

m = TinyTeacher()
concrete = m.serving_default.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], m)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
out = converter.convert()
Path('app/src/main/assets/tiny_teacher.tflite').write_bytes(out)
print('wrote', len(out), 'bytes')
