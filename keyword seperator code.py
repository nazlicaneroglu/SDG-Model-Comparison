import re
import pandas as pd 
# Read the content of the file
with open('C:/Users/pc/Desktop/Thesıs/thesis codes/Aurora queries/query_SDG1.xml', 'r') as file:
    content = file.read()

# Use regular expressions to find words inside quotation marks
keywords = re.findall(r'"([^"]*)"', content)

# Print the extracted keywords
for keyword in keywords:
    print(keyword)
    
df = pd.DataFrame({'Keywords_sdg1': keywords})
df.to_excel('C:/Users/pc/Desktop/Thesıs/thesis codes/rivest et al/rivest SDG16.xlsx', index=False)
