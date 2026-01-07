#!/dl_env/bin/env python3

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
for device in tf.config.list_physical_devices():
    print(f"  {device}")
    if device.device_type == 'GPU':
        print(f"Device: {device.name}")
        print(f"Type: {device.device_type}")
        print(f"  Memory growth enabled: {tf.config.experimental.get_memory_growth(device)}")

tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print(c)