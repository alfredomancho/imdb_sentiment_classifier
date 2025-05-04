import tensorflow as tf

from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import TextVectorization

##########################################
# Write custom review(s)
##########################################
my_reviews = [
        "That movie was complete garbage. Pour bleach on my eyes ffs. My dog could have wrote a script better than that.",
        "Very good movie! I purchased the movie digitally right aterwards.",
        "Nice movie.  Would recommend.",
        "Horrible movie.  I vomited because it was so bad.",
        "This movie could have been better if the acting was decent.",
]

##########################################
# Configuration
##########################################
MODEL_FILE = 'sentiment_model.keras'
#MODEL_FILE = 'sentiment_model.h5'

# Load trained model
sentiment_model = tf.keras.models.load_model(MODEL_FILE)
#sentiment_model = load_model(MODEL_FILE)  


def predict_sentiment(model, texts):
    """
    Return sentiment labels for a list of raw review strings.
    """
    # model now accepts tf.constant of strings directly
    scores = model.predict(tf.constant(texts), verbose=0).flatten()
    return ['Positive' if s > 0.5 else 'Negative' for s in scores], scores


# Make predictions and display to screen
labels, scores = predict_sentiment(sentiment_model, my_reviews)
for rev, lbl, sc in zip(my_reviews, labels, scores):
    print(f"{lbl:>8} ({sc:.3f}): {rev}")
