from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer



pline = pipeline("token-classification", model="Media1129/recipe-tag-model", tokenizer="Media1129/recipe-tag-model", aggregation_strategy='simple')


utterances = ["Thai cuisine is what I want to cook today", "Do I need any ingredients for cheesecake?", "I want to make juice. Can you please teach me?", "Teach me how to make aci bowl", "I want to make spicy spaghetti"]

utterances_dev = ["I need a recipe for vegetarian lasagna", "Please give me a recipe with low calorie. I want to lose weight recently.", "What kind of food is suitable for Christmas?", "Show my some recipe for the Easter's", "I don't have electric mixer. Can you find some chocolate cake recipe without using that?" ,"Guide me to cook something healthy"]



# index start from 1
for sen_idx, utterance in enumerate(utterances):
    outputs = pline(utterance)
    
    dishname = []
    ingredient = []

    for output in outputs:
        if output['entity_group'] == "dishname":
            dishname.append(output['word'])
        elif output['entity_group'] == "ingredient":
            ingredient.append(output['word'])
    print("sentence: {}".format(utterance))
    print("dishname: {}".format(dishname))
    print("ingredient: {}".format(ingredient))
    print()








# Ground truth
# Thai cuisine is what I want to cook today
# B-dishname I-dishname O O O O O O O

# Do I need any ingredients for cheesecake?
# O O O O O O B-dishname

# I want to make juice. Can you please teach me?
# O O O O B-dishname O O O O O

# Teach me how to make aci bowl
# O O O O O B-dishname I-dishname

# I want to make spicy spaghetti
# O O O O B-dishname I-dishname



# Pipeline predict



# model predict
# B-dishname I-dishname O O O O O O B-dishname
# O O O O O O B-dishname
# O O O O B-dishname O O O O O
# O O O O O B-dishname I-dishname
# O O O O B-dishname I-dishname








