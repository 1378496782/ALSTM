import json

def extract_lstm_model(notebook_file):
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def create_model' in source or 'model = Sequential' in source or 'LSTM(' in source:
                print(f"Found LSTM model code in {notebook_file}:")
                print(source)
                print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    files = ["ALSTM1.ipynb", "demo1.ipynb", "网络流量预测-LSTM.ipynb"]
    for file in files:
        try:
            extract_lstm_model(file)
        except Exception as e:
            print(f"Error processing {file}: {e}") 