from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm






quora_path = "quora/baking.txt" # 824
# quora_path = "quora/Cooking-Tips-and-Hacks.txt" # 816
# quora_path = "quora/cooking.txt" # 754
# quora_path = "quora/desserts.txt" # 645
# quora_path = "quora/Food.txt" # 888
# quora_path = "quora/healthy-cooking.txt" # 518
# quora_path = "quora/homemade-items.txt" # 468
# quora_path = "quora/homemade-recipes.txt"# 73
# quora_path = "quora/learning-to-cook.txt" # 734
# quora_path = "quora/meat.txt" # 686
# quora_path = "quora/recipes.txt" # 497




utterances = []
dishnames_first_col = []
dishnames_second_col = []
dishnames_third_col = []

ingredients_first_col = []
ingredients_second_col = []
ingredients_third_col = []



with open(quora_path, 'r') as f:
    for line in f.readlines():
        utterances.append(line.rstrip())


pline = pipeline(
    "token-classification", 
    model="Media1129/recipe-tag-model", 
    tokenizer="Media1129/recipe-tag-model", 
    aggregation_strategy='simple'
)


for sen_idx, utterance in enumerate(tqdm(utterances)):
    # print("sentence index: {}".format(sen_idx))
    outputs = pline(utterance)
    # print(outputs)
    
    dishname = []
    dishname_score = []
    ingredient = []
    ingredient_score = []


    for output in outputs:
        if output['entity_group'] == "dishname":
            dishname.append(output['word'])
            dishname_score.append("{:.2f}".format(output['score']))
        elif output['entity_group'] == "ingredient":
            ingredient.append(output['word'])
            ingredient_score.append("{:.2f}".format(output['score']))
    
    if len(dishname) >= 3:
        dishnames_first_col.append(dishname[0]+" ({})".format(dishname_score[0]))
        dishnames_second_col.append(dishname[1]+" ({})".format(dishname_score[1]))
        dishnames_third_col.append(dishname[2]+" ({})".format(dishname_score[2]))
    elif len(dishname) == 2:
        dishnames_first_col.append(dishname[0]+" ({})".format(dishname_score[0]))
        dishnames_second_col.append(dishname[1]+" ({})".format(dishname_score[1]))
        dishnames_third_col.append("none")
    elif len(dishname) == 1:
        dishnames_first_col.append(dishname[0]+" ({})".format(dishname_score[0]))
        dishnames_second_col.append("none")
        dishnames_third_col.append("none")
    elif len(dishname) == 0:
        dishnames_first_col.append("none")
        dishnames_second_col.append("none")
        dishnames_third_col.append("none")

    if len(ingredient) >= 3:
        ingredients_first_col.append(ingredient[0]+" ({})".format(ingredient_score[0]))
        ingredients_second_col.append(ingredient[1]+" ({})".format(ingredient_score[1]))
        ingredients_third_col.append(ingredient[2]+" ({})".format(ingredient_score[2]))
    elif len(ingredient) == 2:
        ingredients_first_col.append(ingredient[0]+" ({})".format(ingredient_score[0]))
        ingredients_second_col.append(ingredient[1]+" ({})".format(ingredient_score[1]))
        ingredients_third_col.append("none")
    elif len(ingredient) == 1:
        ingredients_first_col.append(ingredient[0]+" ({})".format(ingredient_score[0]))
        ingredients_second_col.append("none")
        ingredients_third_col.append("none")
    elif len(ingredient) == 0:
        ingredients_first_col.append("none")
        ingredients_second_col.append("none")
        ingredients_third_col.append("none")






df = pd.DataFrame(
    {
        'utterance': utterances,
        'dishname_first (score)': dishnames_first_col,
        'dishname_second (score)': dishnames_second_col,
        'dishname_third (score)': dishnames_third_col,
        'ingredient_first (score)': ingredients_first_col,
        'ingredient_second (score)': ingredients_second_col,
        'ingredient_third (score)': ingredients_third_col
    })

f_name = quora_path.split('.')[0].split('/')[1]
df.to_csv('quora_predict/{}.csv'.format(f_name), index=False)

