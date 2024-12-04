from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import json

# Define LLMS models
llms_models = ['llama3.1', 'phi3.5', 'gemma2']
entities = []
output = []

text = """President Joe Biden criticized the Supreme Court's decision on the redistricting of the South Carolina district by issuing a public statement highlighting concerns about racial discrimination."""

system_prompt = """
You are a helpful information extraction system. Consider the following criterias:
1. DO  not use acronyms to refer to entities.
2. Only output entities that are in the passage (i.e., keep the exact text of the entity).
4. if you detect two entities together, consider them as one entity. (i.e., text: Colombia, Barranquilla; entity: Colombia, Barranquilla; type: Location)
"""

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

# Define the input text and prompt
prompt = f"Given a passage, your task is to extract all entities and identify their entity types. The output should be a in a list of tuples of the following format: [(\"entity 1\", \"type of entity 1\"),..]\n Passage: {text}"


for model in llms_models:
    chat_model = Ollama(model=model)
    response = chat_model.invoke([prompt])
    print(f'model: {model}')
    print(response)
    print('---')

    # Regex pattern to extract entities and their types
    pattern = r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'

    # Find all matches in the string
    matches = re.findall(pattern, response)

    # Reset entities list for each model
    entities = []

    # Print the extracted entities and their types
    for entity, entity_type in matches:
        entities.append({'entity': entity, 'type': entity_type})
    
    output.append({'model': model, 'response': entities})

for model_output in output:
    print(f"Model: {model_output['model']}")
    for entity in model_output['response']:
        print(f"Entity: {entity['entity']}, Type: {entity['type']}")
    print('---')

# Save the final list of entities by model into a JSON file
output_file_path = os.path.join("utils", "extracted_entities.json")

# Write the data to the JSON file
with open(f"extracted_entities.json", 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

    
