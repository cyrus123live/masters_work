import json

FILENAME = "stock_data/APPL_fun.json"

with open(FILENAME, 'r') as f:
    file_string = ""
    for line in f.readlines():
        file_string += line.strip()
    fun = json.loads(file_string)
    

for h in fun["Financials"]:
    print(h)
    # print(fun[h])

print(json.dumps(fun["Earnings"]["History"], indent=2)) # Historical earnings actual and estimate
print(json.dumps(fun["Financials"]["Balance_Sheet"], indent=2)) # Yearly balance sheets 
print(json.dumps(fun["Financials"]["Cash_Flow"], indent=2)) # yearly cash flow
print(json.dumps(fun["Financials"]["Income_Statement"], indent=2)) # yearly Income statements