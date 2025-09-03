import json

result_path = '/home/mixlab/tabular/medthink/output/rad_closed_end_generate/Explanation/First/test_response.json'
label_path = '/home/mixlab/tabular/medthink/Medthink/Medthink_Dataset/R-RAD/closed-end/testset.json'

with open(result_path, 'r') as f:
    results = json.load(f)

with open(label_path, 'r') as f:
    labels = json.load(f)

option_to_id = {'A': 0, 'B': 1}
correct = 0
total = 0

for id_with_prefix, prediction in results.items():
    id = id_with_prefix.split("_")[1]
    true_option = labels[id]["answer"]
    
    before_solution = prediction.split("Solution")[0]

    if "(" in prediction and ")" in prediction:
        pred_option = before_solution.split("(")[-1].split(")")[0].strip()
        pred_label = option_to_id.get(pred_option)
    else:
        continue  

    total += 1
    if pred_label == true_option:
        correct += 1

# 5. 정확도 계산
accuracy = correct / total if total > 0 else 0.0
print('total:', total)
print('correct:', correct)
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
