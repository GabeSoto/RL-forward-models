import tensorflow as tf
#import serial
import numpy as np
import time
from Forward_Model import Normalizer, ForwardModel
from Prac_Env import prac_env_v0
from logamp_main import Params
from gym import spaces


# Load trained RL model
params = Params()
env = prac_env_v0()

optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
model = ForwardModel(output_units=env.observation_space.shape[-1])
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

load_ckpt = tf.train.latest_checkpoint('.\logs')
checkpoint.restore(load_ckpt)


#Pick port
# ser = serial.Serial('COM3', 9600, timeout=1)


# #Get input
# def get_sensor_data():
#     # Read from your hardware or sensors (example: read one line from serial)
#     ser.write(b"GET_STATE\n")
#     line = ser.readline().decode().strip()
#     values = [float(x) for x in line.split(",")]
#     return np.array(values).reshape(1, -1)  # Reshape for batch input


#Send output
# def send_current(current_value):
#     command = f"SET_CURRENT:{current_value:.3f}\n"
#     ser.write(command.encode())


#Predict new currents for the system
target = 200.0
current = -289.0

action_space = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100,
                -0.0001, -0.0003, -0.001, -0.003, -0.01, -0.03, -0.1, -0.3,  -1, -3, -10, -30, -100]



try:
    while True:
        # Step 1: Get input/state
        #state = get_sensor_data()  # Shape: (1, num_features)


        # Step 2: Predict control signal (e.g., current)
        #predicted = load_ckpt.predict(state)  # Shape: (1, output_dim)
        target += model(current, action_space)  # Example prediction

        # Step 3: Send output to system
        #current_value = float(predicted[0][0])  # Or apply scaling/clipping
        print(f"Predicted current: {current}")
        #send_current(current_value)

        # Step 4: Optional delay (e.g., 100 Hz loop)
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopped by user.")

# finally:
#     ser.close()