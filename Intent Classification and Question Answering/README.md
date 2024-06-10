# Intent Classification and Question Answering

Intent classification and question answering are crucial tasks in natural language processing and information retrieval. Intent classification involves categorizing user queries based on their intended meaning, which helps in providing accurate and efficient responses. In this exercise, we aim to implement an intent classification model based on the details provided in the article, followed by utilizing LSTM architectures for answering questions.

## Model Architecture

### Intent Classification:
- **Architecture**: The proposed architecture involves using embedding and LSTM layers. Embedding layers convert words into numerical vectors, and LSTM layers process sequential data to capture long-term dependencies.
- **Reason for Using LSTM**: LSTM architecture is chosen for its ability to handle sequence data effectively, capturing contextual information and handling vanishing gradient problems better than traditional RNNs.
- **Word Embeddings**: GloVe embeddings are preferred for words with multiple meanings due to their ability to capture semantic relationships and word contexts effectively.

## Data Preprocessing

### Preprocessing Data:
- **Tokenization**: Data is tokenized to convert text into individual tokens or words.
- **Normalization**: Text normalization techniques are applied to standardize the text, such as converting to lowercase and removing punctuation.

## Implementation of Intent Classification

### Model Implementation:
- Two models are proposed for intent classification: one predicts only the top-level category, and the other predicts both levels of categories.
- Both models are implemented and trained using the provided dataset.
- Confusion matrices, accuracy, precision, recall, F1 score, and other evaluation metrics are calculated and analyzed over time.

## Responder Model Implementation

### Responder Model:
- The responder model, based on the proposed architecture, is implemented to provide relevant answers to the given questions.
- The model is trained on the question-answer dataset.
- The model is tested on sample questions provided in the QA_data file, and the responses are reported.

### Sample Questions for Testing:
- "How many people speak French?"
- "What day is today?"
- "Who will win the war?"
- "Who is the Italian first minister?"
- "When did World War II end?"
- "When was Gandhi assassinated?"

## Conclusion

In conclusion, intent classification and question answering are essential tasks in natural language understanding. Implementing models based on LSTM architectures and GloVe embeddings can effectively classify user intents and provide relevant answers to questions. Evaluation metrics provide insights into model performance, aiding in further refinement and improvement.