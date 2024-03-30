# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import tensorflow as tf
import numpy as np
from node_cell import (
    LSTMCell,
    CTRNNCell,
    ODELSTM,
    VanillaRNN,
    CTGRU,
    BidirectionalRNN,
    GRUD,
    PhasedLSTM,
    GRUODE,
    HawkLSTMCell,
)
import argparse
from irregular_sampled_datasets import Walker2dImitationData

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    time = []
    input = []
    output = []
    collect_data = False
    all_time = []
    all_input = []
    all_output = []
    min_length = 0

    for line in lines:
        if line.startswith('Step Information'):
            if time != []:
                all_time.append(time)
                all_input.append(input)
                all_output.append(output)
                if min_length == 0 or len(time) < min_length:
                    min_length = len(time)
            collect_data = True
        elif line.strip() and collect_data:
            parts = line.split()
            time.append([float(parts[0])])
            input.append([float(parts[1])])
            output.append([float(parts[2])])
        elif 'time' in line and 'V(1)' in line and 'V(20)' in line:
            collect_data = True
            
        all_time_final = []
        all_input_final = []
        all_output_final = []
            
        for time_aux, input_aux, output_aux in all_time, all_input, all_output:
            if len(time_aux) > min_length:
                all_time_final.time_aux[:min_length]
                all_input_final.input_aux[:min_length]
                all_output_final.output_aux[:min_length]

    return all_time_final, all_input_final, all_output_final, min_length

def reshape_inputs(inputs, time_steps, features):
    # Assuming `inputs` is a flat list of floats, we reshape it into [samples, time_steps, features]
    # This might involve splitting your continuous data into sequences.
    sample_length = time_steps * features
    num_samples = len(inputs) // sample_length
    reshaped = np.array(inputs[:num_samples * sample_length]).reshape(num_samples, time_steps, features)
    return reshaped


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.005, type=float)
args = parser.parse_args()

time, inputs, targets, seq_length = load_data('./circuits/ADVANCED_AMPLIFIER2.txt')
input_size = 1
        
if not time or not inputs or not targets:
    raise ValueError("No data loaded. Check the file path and file format.")

data = Walker2dImitationData(seq_len=64)

if args.model == "lstm":
    cell = LSTMCell(units=args.size)
elif args.model == "ctrnn":
    cell = CTRNNCell(units=args.size, num_unfolds=3, method="rk4")
elif args.model == "node":
    cell = CTRNNCell(units=args.size, num_unfolds=3, method="rk4", tau=0)
elif args.model == "odelstm":
    cell = ODELSTM(units=args.size)
elif args.model == "ctgru":
    cell = CTGRU(units=args.size)
elif args.model == "vanilla":
    cell = VanillaRNN(units=args.size)
elif args.model == "bidirect":
    cell = BidirectionalRNN(units=args.size)
elif args.model == "grud":
    cell = GRUD(units=args.size)
elif args.model == "phased":
    cell = PhasedLSTM(units=args.size)
elif args.model == "gruode":
    cell = GRUODE(units=args.size)
elif args.model == "hawk":
    cell = HawkLSTMCell(units=args.size)
else:
    raise ValueError("Unknown model type '{}'".format(args.model))

signal_input = tf.keras.Input(shape=(seq_length, input_size), name="robot")
time_input = tf.keras.Input(shape=(seq_length, 1), name="time")

rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)

output_states = rnn((signal_input, time_input))
y = tf.keras.layers.Dense(input_size)(output_states)

model = tf.keras.Model(inputs=[signal_input, time_input], outputs=[y])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss=tf.keras.losses.MeanSquaredError(),
)
model.summary()

hist = model.fit(
    x=(data.train_x, data.train_times),
    y=data.train_y,
    batch_size=128,
    epochs=args.epochs,
    validation_data=((data.valid_x, data.valid_times), data.valid_y),
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            "/tmp/checkpoint", save_best_only=True, save_weights_only=True, mode="min"
        )
    ],
)

# Restore checkpoint with lowest validation MSE
model.load_weights("/tmp/checkpoint")
best_test_loss = model.evaluate(
    x=(data.test_x, data.test_times), y=data.test_y, verbose=2
)
print("Best test loss: {:0.3f}".format(best_test_loss))

# Log result in file
base_path = "results/walker"
os.makedirs(base_path, exist_ok=True)
with open("{}/{}_{}.csv".format(base_path, args.model, args.size), "a") as f:
    f.write("{:06f}\n".format(best_test_loss))