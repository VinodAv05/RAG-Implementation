import nest_asyncio
from llama_parse import LlamaParse
import os

from RAG_implementaion import getDataFromJSONFile

getLLAMAKey = getDataFromJSONFile("config.JSON","LLAMA_CLOUD_API_KEY")
os.environ["LLAMA_CLOUD_API_KEY"]=getLLAMAKey

nest_asyncio.apply()

def convertToMarkDown(fileName):
    converted_doc = LlamaParse(result_type="markdown").load_data(fileName)
    print("Document converted for=>", fileName)
    return converted_doc


def writeToNewFile(fileName, documentContent):
    with open(fileName.split(".")[0] + ".md", "w") as file:
        for i in range(len(documentContent)):
            file.write(documentContent[i].text)
            file.write("\n")
        print("MD Document created for=>", fileName)