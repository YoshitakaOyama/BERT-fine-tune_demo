[Reference](https://towardsdatascience.com/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a)

# Execution procedure
1. Install required python package.
2. We will be using the [Stanford Question Answering Dataset (SQuAD 2.0)](https://rajpurkar.github.io/SQuAD-explorer/) for training and evaluating our model. SQuAD is a reading comprehension dataset and a standard benchmark for QA models. The dataset is publicly available on the website. Download the dataset and place the files (train-v2.0.json, dev-v2.0.json) in the data/ directory.
3. If you want to train model, run
```
python demo.py
```
When training a model, I highly recommend checking out all the options [here](https://github.com/ThilinaRajapakse/simpletransformers/blob/master/README.md#default-settings).
Upon completion of training, the final model will be saved to `./outputs` dir.

When downloading the model, please download [here](#).

4. To do Question Answering,　Pass the question list and reference string to the `answer_questions function` in `predict.py` and execute it. The answer comes back. The model and tokenizer used here are automatically downloaded at the first startup.

ex.)
```
import textwrap
from bert.predict import answer_questions


wrapper = textwrap.TextWrapper(width=80)

bert_abstract = 'European Union law is a body of treaties and legislation, such as Regulations and Directives, which have direct effect or indirect effect on the laws of European Union member states. The three sources of European Union law are primary law, secondary law and supplementary law. The main sources of primary law are the Treaties establishing the European Union. Secondary sources include regulations and directives which are based on the Treaties. The legislature of the European Union is principally composed of the European Parliament and the Council of the European Union, which under the Treaties may establish secondary law to pursue the objective set out in the Treaties.'

# Be sure to enter the id
questions = [
    {
        'id': 1,
        'question': 'What are the three sources of European Union law?'
    },
    {
        'id': 2,
        'question': 'What is European Union Law?'
    },
    {
        'id': 3,
        'question': 'What are the main sources of primary law?'
    },
    {
        'id': 4,
        'question': 'What are the secondary sources of primary law?'
    }
]


answers = answer_questions(questions, bert_abstract)
print('-'*60)
print(f"bert_abstract(reference): \n{wrapper.fill(bert_abstract)}\n")
for question, answer in zip(questions, answers):
    print(f"question: {question['question']}")
    print(f"answer: {answer['answer']}\n")
```

