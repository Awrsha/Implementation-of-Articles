# Question-Answering System

In this exercise, we will delve into the realm of question-answering systems, a popular subfield of natural language processing, and practice building one.

## Extractive QA System

Recent advances in natural language processing, particularly with the advent of models like BERT (Bidirectional Encoder Representations from Transformers), have significantly improved the performance of question-answering systems. These systems aim to automatically extract answers from a given text, playing vital roles in various domains such as information retrieval, conversational agents, and virtual assistants.

The Extractive QA task involves designing and implementing a system based on transformer models, such as BERT, to extract the best answer from a text given a question. Unlike generative QA systems that generate answers from scratch, extractive QA systems rely on information present in the text.

## Model Architecture

### Modeling the Problem:
We start by understanding the original BERT model proposed by Devlin et al. (2019) and its pre-training objectives. Then, we design the overall structure of our model tailored specifically for this task, including input, output, model architecture, loss functions, and what the model is supposed to learn during training.

## Data Preprocessing

### Preprocessing Data:
The dataset used for this exercise is the PQuAD (Persian Question Answering Dataset) provided by Darvishi et al. (2023). We first explore the dataset, visualize some statistics, and then perform the necessary preprocessing steps required for the task, such as tokenization and normalization.

## Model Implementation

### Implementing the Model:
For model implementation, we have two pre-trained models available: ParsBERT and ALBERT. Both are based on the BERT architecture and have been fine-tuned for Persian language understanding. We utilize the Hugging Face library to implement our models based on these pre-trained architectures.

## Evaluation and Postprocessing

### Evaluation and Postprocessing:
During the task execution, we need to handle exceptions, preprocess and postprocess data to manage the limitations of transformer models regarding input length. After managing exceptions, we evaluate the performance of our trained models on the test dataset using evaluation metrics such as Exact Match (EM) and F1-score, comparing our results with those reported in the literature.

## Conclusion

This exercise provides hands-on experience in building an extractive question-answering system using transformer-based models like BERT. By following the steps outlined above, we gain insights into the complexities of NLP tasks and the practical challenges involved in implementing state-of-the-art models for real-world applications.

## References

1. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" NAACL 2019. [Link](https://doi.org/10.18653/v1/N191423)
2. Darvishi, K. et al. (2023). "PQuAD: A Persian question answering dataset," Computer Speech & Language, 80, p. 101486. [Link](https://doi.org/10.1016/j.csl.2023.101486)
3. Farahani, M., Gharachorloo, M., Farahani, M. et al. (2021). "ParsBERT: Transformer-based Model for Persian Language Understanding." Neural Process Lett 53, 3831–3847. [Link](https://doi.org/10.1007/s11063-021-10528-4)

## Additional Resources
- [PQuAD Dataset](https://github.com/AUT-NLP/PQuAD)
- [ParsBERT](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased)
- [ALBERT](https://huggingface.co/m3hrdadfi/albert-fa-base-v2)