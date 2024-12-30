import json
import os
import urllib.request

file_path = "instruction-data.json"
with open(file_path, "r") as file:
    data = json.load(file)
    
def format_input(entry, format_style="alpaca"):
    """_summary_

    Args:
        entry (_type_): dictionary of input data.
        format_style (str, optional): Prompt style: "alpaca" or "phi". Defaults to "alpaca".
    """
    
    if format_style == "alpaca":
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that approriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        
        input_text = (
            f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        )
        return instruction_text + input_text
    
    elif format_style == "phi":
        instruction_text = (
            f"<|user|>\n{entry['instruction']}: "
        )
        
        input_text = (
            f"{entry['input']}" if entry["input"] else ""
        )
        return instruction_text + input_text
    
    else:
        raise ValueError(f"Invalid format style: {format_style}")
    
# data_idx = 999 
# model_input = format_input(data[data_idx], format_style="alpaca")
# desired_response = f"\n\n### Response:\n{data[data_idx]['output']}" # For the alpaca format only
# print(model_input + desired_response)

# train_portion = int(len(data) * 0.85)
# test_portion = int(len(data) * 0.1)
# val_portion = len(data) - train_portion - test_portion

# train_data = data[:train_portion]
# test_data = data[train_portion:train_portion + test_portion]
# val_data = data[train_portion + test_portion:]

# print("Train data size:", len(train_data))
# print("Test data size:", len(test_data))
# print("Validation data size:", len(val_data))



