from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer



pline = pipeline("token-classification", model="adamlin/recipe-tag-model", tokenizer="adamlin/recipe-tag-model", aggregation_strategy='simple')

# utterance1 = "Thai cuisine is what I want to cook today"
# utterance2 = "Do I need any ingredients for cheesecake?"
# utterance3 = "I want to make juice. Can you please teach me?"
# utterance4 = "Teach me how to make aci bowl"
# utterance5 = "I want to make spicy spaghetti"

utterances = ["Thai cuisine is what I want to cook today", "Do I need any ingredients for cheesecake?", "I want to make juice. Can you please teach me?", "Teach me how to make aci bowl", "I want to make spicy spaghetti"]
labels = ["O", "dishname", "ingredient"]


# index start from 1
for sen_idx, utterance in enumerate(utterances):
    # print("sentence num: {}".format(sen_idx))
    outputs = pline(utterance)
    # print(outputs)
    # print(outputs)
    # break
    
    dishname = []
    ingredient = []

    input_text = utterance
    input_list = input_text.split()
    for output in outputs:
        if output['entity_group'] == "dishname":
            dishname.append(output['word'])
        elif output['entity_group'] == "ingredient":
            ingredient.append(output['word'])
    print(dishname)
    print(ingredient)
    
    # break


# Pipeline predict
# dishname dishname O O O O O O O
# O O O O O O dishname dishname O
# O O O O dishname O O O O O O O
# O O O O O dishname dishname dishname
# O O O O dishname dishname


# model predict
# ['dishname', 'dishname', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# ['O', 'O', 'O', 'O', 'O', 'O', 'dishname']
# ['O', 'O', 'O', 'O', 'dishname', 'O', 'O', 'O', 'O', 'O']
# ['O', 'O', 'O', 'O', 'O', 'dishname', 'dishname']
# ['O', 'O', 'O', 'O', 'dishname', 'dishname']



