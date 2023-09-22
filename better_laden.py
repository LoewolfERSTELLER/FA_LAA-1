import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Lade das gespeicherte Modell, den Tokenizer und die maximale Sequenzlänge
model = tf.keras.models.load_model("FA_LAA-1/text_model.h5")
with open('FA_LAA-1/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('FA_LAA-1/max_len.txt', 'r') as f:
    MAX_LEN = int(f.read())

def get_answer(question):
    """
    Get the model's answer.
    
    Parameters:
    - question: the current question asked by the user.
    
    Returns:
    - answer: the model's response.
    """
    seq_question = tokenizer.texts_to_sequences([question])
    seq_question = pad_sequences(seq_question, maxlen=MAX_LEN, padding='post')
    
    prediction = model.predict(seq_question)
    predicted_sequence = [int(tf.argmax(token, axis=0)) for token in prediction[0]]
    
    return tokenizer.sequences_to_texts([predicted_sequence])[0].replace(' <OOV>', '')

def multi_task_answer(multi_question):
    """
    Split the multi-question string into individual questions and get answers for each.
    
    Parameters:
    - multi_question: string containing multiple questions separated by specific delimiters.
    
    Returns:
    - combined_answer: combined answer for all the questions.
    """
    questions = multi_question.split(".")
    questions = multi_question.split("?")
    questions = multi_question.split("!")
   # Assuming questions are separated by semicolons
    answers = [get_answer(q.strip()) for q in questions]  # Get the answer for each question
    combined_answer = " ".join(answers)
    
    return combined_answer

# Interaction with the user considering multi-tasking
while True:
    frage = input("Frage stellen (oder 'exit' zum Beenden): ")
    if frage.lower() == 'exit':
        break
    answer = multi_task_answer(frage)
    print(answer)
