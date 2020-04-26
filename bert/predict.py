import json
import os
from pprint import pprint
import textwrap
from simpletransformers.question_answering import QuestionAnsweringModel


def answer_questions(questions, answer_text, model_path=None, use_cuda=False):
    """
    Takes a `question` dict list and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer.

    Parameters
    ----------
    questions : list
    ex.)
        questions = [
        {
            'id': 1,
            'question': 'What are the three sources of European Union law?'
        },
        ...
    ]

    answer_text : str
    model_path : str
    use_cuda: bool

    Return
    -------
    preds : list
    ex.)
        [
            {
                'id': '1',
                'answer': 'primary law, secondary law and supplementary law'
            },
            ...
        ]
    """
    if model_path is None:
        model_path = os.path.dirname(os.path.abspath(__file__)) + '/outputs'

    # model = QuestionAnsweringModel('bert', './outputs/final', use_cuda=use_cuda)
    model = QuestionAnsweringModel('bert', model_path, use_cuda=use_cuda)
    data_to_predict = [
        {
            'context': answer_text,
            'qas': questions
        }
    ]

    preds = model.predict(data_to_predict)
    return preds


if __name__ == '__main__':
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
