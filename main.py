# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:33:07 2023

@author: pc
"""

import openai
import pandas as pd

#Get data
dataset= pd.read_excel('C:/Users/pc/Desktop/chatgpt/chatgpt/reduceddata.xlsx')

# Set up the OpenAI API client
openai.api_key = ""

# Then, you can call the "gpt-3.5-turbo" model
model_engine = "gpt-3.5-turbo"

# set your input text
pretext = "can you tell me the most prominent sdgs in this text  Please just answer with numbers of the goals no other text only the numbers "
question = "Indeed, there is no guarantee that results on the capacity of a social programme to reduce poverty by a given amount will translate to another country if the poverty line is set in a different way. We propose that this line be set at a multiple of median income, which makes the poverty line sensitive to the distribution of welfare in the country, rather than being solely determined by the mean of the welfare metric used. This relative poverty line can fruitfully be used alongside an absolute poverty line, which we set at the level of the international dollar-a-day poverty line. Considering that the international 1.25 dollar-a-day poverty line sets a minimum income for fulfilling survival needs, the 1.25 dollar-a-day line may appear more relevant when the relative poverty line falls below this level."
#post_text = " Please just answer with numbers of the goals no other text only the numbers for example your answer should be 1,2 if the goal 1 and goal 2 are the most prominent"
input_text = pretext + question

answers=[]
for x in range(0, 11):
    question = str(dataset.iloc[x])
    input_text = pretext + question
    print(input_text)
    try:
        
        response = openai.ChatCompletion.create(
           model=model_engine,
           messages=[{"role": "user", "content": input_text }]
        )
        answers.append(response)
    except:
        continue
    
# Send an API request and get a response, note that the interface and parameters have changed compared to the old model
response = openai.ChatCompletion.create(
   model=model_engine,
   messages=[{"role": "user", "content": input_text }]
)


# Parse the response and output the result
output_text = response['choices'][0]['message']['content']
real_answers = []
print("ChatGPT API reply:", output_text)
for x in answers:    
    print(x['choices'][0]['message']['content'])
    print()
    real_answers.append(x['choices'][0]['message']['content'])
    
df = pd.DataFrame(real_answers, columns=['answers'])
df.to_csv('answers.csv', index=False)
