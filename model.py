#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow.keras import layers

# print Tensorflow version
print(tf.__version__)

# =====================
# Configuration
# =====================
VOCAB_SIZE = 20000       # Max vocabulary size
SEQ_LENGTH = 250         # Length to pad or truncate each review
EMBEDDING_DIM = 16       # Embedding output dimension
BATCH_SIZE = 32          # Samples per batch
EPOCHS = 5               # Number of training passes
MODEL_PATH = 'sentiment_model.keras'
LOSS_PLOT_PATH = 'loss_curves.png'


def build_sentiment_model(text_vectorizer):
    """
    Build a model that accepts raw strings, vectorizes them,
    embeds, pools, and classifies.
    """
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name='raw_review')
    x      = text_vectorizer(inputs)
    x      = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,
                              name='embed')(x)
    x      = layers.GlobalAveragePooling1D(name='avg_pool')(x)
    x      = layers.Dense(16, activation='relu', name='hidden')(x)
    outputs= layers.Dense(1, activation='sigmoid', name='prediction')(x)

    model = tf.keras.Model(inputs, outputs, name='imdb_sentiment')
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_store(model, train_data, test_data):
    """
    Train model, save to disk, and return the training history.
    """
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data
    )
    model.save(MODEL_PATH)
    return history


def evaluate_model(model, test_data):
    """
    Evaluate model performance on test set.
    """
    loss, accuracy = model.evaluate(test_data)
    print(f'Test Loss: {loss:.4f}  Test Accuracy: {accuracy:.4f}')


def plot_loss(history):
    """
    Plot training and validation loss curves and save to file.
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    print(f'Loss curves saved to {LOSS_PLOT_PATH}')


def predict_sentiment(model, texts):
    """
    Return sentiment labels for a list of raw review strings.
    """
    # model now accepts tf.constant of strings directly
    scores = model.predict(tf.constant(texts), verbose=0).flatten()
    return ['Positive' if s > 0.5 else 'Negative' for s in scores], scores


if __name__ == '__main__':
    
    # Load raw IMDB dataset
    raw_train, raw_test = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        as_supervised=True
    )

    # Create & adapt the vectorizer on raw text
    text_vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQ_LENGTH,
        name='vectorizer'
    )
    text_vectorizer.adapt(raw_train.map(lambda text, _: text))

    # Batch datasets 
    train_ds = (raw_train
                .shuffle(10000)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    test_ds  = (raw_test
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    # Build model
    sentiment_model = build_sentiment_model(text_vectorizer)
    sentiment_model.summary()

    # Train and retrieve history
    history = train_and_store(sentiment_model, train_ds, test_ds)

    # Evaluate on test set
    evaluate_model(sentiment_model, test_ds)

    # Plot and save loss curves
    plot_loss(history)
