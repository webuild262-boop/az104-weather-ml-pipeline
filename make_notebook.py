import json
with open("az104_data_cleaner.py", encoding="utf-8") as f:
    src = f.read()
nb = {"nbformat":4,"nbformat_minor":5,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.14.0"}},
      "cells":[{"cell_type":"code","execution_count":None,"id":"c0",
                "metadata":{},"outputs":[],"source":src}]}
with open("az104_data_cleaning.ipynb","w",encoding="utf-8") as f:
    json.dump(nb,f,indent=1)
print("Done! Notebook created.")
