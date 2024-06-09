import os
import pickle  # importing the libraries
import numpy  # importing the numpy library

from music21 import instrument, note, stream, chord  # importing the music library
from keras.models import Sequential  # importing the sequential library
from keras.layers import Dense  # importing the dense library
from keras.layers import Dropout  # importing the Dropout library
from keras.layers import LSTM  # importing the LSTM library
from keras.layers import BatchNormalization as BatchNorm  # importing the batch normalization function
from keras.layers import Activation  # importing the activation function


def generate(): #here we are defining a function
    with open('data/notes', 'rb') as filepath:#opening a file
        notes = pickle.load(filepath)#loading the file using pickle
 

    pitchnames = sorted(set(item for item in notes))  # using the sorted function to sort the items in file

    n_vocab = len(set(notes))  # counting the length of the notes.

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)  # here we are preparing the sequence for the notes.
    model = create_network(normalized_input, n_vocab)  # here we are creating the networks
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)  # here we are generating the notes using the model.
    create_midi(prediction_output)  # here we are calling the create_midi function


def prepare_sequences(notes, pitchnames, n_vocab):  # defining the function
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))  # creating the dictionary

    seq_len = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - seq_len, 1):
        seq_in = notes[i:i + seq_len]
        seq_out = notes[i + seq_len]
        network_input.append([note_to_int[char] for char in seq_in])
        output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    normalized_input = numpy.reshape(network_input, (n_patterns, seq_len, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def create_network(network_input, vocab_size):
    model = Sequential()  # creating the model
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))  # adding layers and features to our model.
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))  # adding LSTM Layer
    model.add(LSTM(512))  # adding LSTM Layer

    model.add(BatchNorm())  # adding batch norm layer
    model.add(Dropout(0.3))  # adding dropout layer
    model.add(Dense(256))  # adding dense layer
    model.add(Activation('relu'))  # adding activation layer
    model.add(BatchNorm())  # adding batch norm layer
    model.add(Dropout(0.3))  # adding dropout layer
    model.add(Dense(vocab_size))  # adding dense layer
    model.add(Activation('softmax'))  # adding activation softmax function
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')  # compiling the model

    model.load_weights('weights.hdf5')  # loading the weights

    return model


def generate_notes(model, network_input, pitchnames, n_vocab):
    start = numpy.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')


if __name__ == '__main__':  # calling the function
    generate()  # calling the generate function
