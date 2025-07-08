# Sanskrit to English Translation

This project aims to translate Sanskrit text to English using a sequence-to-sequence model with attention mechanism. The dataset used for training is extracted from the `geeta.txt` file, which contains pairs of Sanskrit and English sentences.

## Dataset

The `geeta.txt` file contains Sanskrit sentences and their corresponding English translations. The sentences are alternated, with Sanskrit sentences on even lines and English sentences on odd lines.

## Model Architecture

The model is built using TensorFlow and Keras. It consists of:

- An encoder with a bidirectional LSTM layer.
- A decoder with an LSTM layer.
- An attention mechanism to focus on relevant parts of the input sequence during translation.

## Training

The model is trained on the dataset with the following steps:

1. **Tokenization**: The Sanskrit and English sentences are tokenized and converted to sequences of integers.
2. **Padding**: The sequences are padded to ensure uniform length.
3. **Model Training**: The model is trained using the padded sequences, with a validation split to monitor performance.

## Issue: Overfitting

During training, the model achieved high accuracy on the training data but failed to generalize well to new sentences. This is a classic case of overfitting, where the model learns the training data too well but does not perform well on unseen data.

## Example Translation

Here is an example of the model's translation output:

```python
example_sentence = "धर्मक्षेत्रे कुरुक्षेत्रे"
print("Translated:", translate_sentence(example_sentence))
```

Output:

```
Translated: the said o king of the the the son of the kurus the and the the the the the and in great in great end of gavalgana came end to the sun of the end the the the end each presence of the celestial of  end were like the side with was great of the kurus with was the by
```

As seen, the output is not coherent and does not match the expected translation. This indicates that the model has overfitted to the training data and is generating random outputs for new sentences.

## Conclusion

While the model demonstrates the ability to learn from the dataset, it requires further tuning and possibly more data to generalize better. Techniques such as regularization, dropout, and data augmentation could be explored to mitigate overfitting.
