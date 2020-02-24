import json
import matplotlib.pyplot as plt
import numpy as np

for i in range(50):
    with open('./dfdc_train_part_'+str(i)+'/processed_dataset.json','r') as f:
        data = json.load(f)
    if i==0:
        final_data = data
    else:
        final_data.update(data)

with open('./processed_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)


