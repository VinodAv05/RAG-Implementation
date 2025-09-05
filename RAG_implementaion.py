import chromadb
import google.generativeai as genai
import json

CHROMA_DATA_PATH = "chroma_db/"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def QueryChromas(query,CollectionName,RetrurnResults=1,SearchOutsideDB=False):
    collection = client.get_collection(name=CollectionName)
    results = collection.query(
        query_texts=[query],
        n_results=RetrurnResults,
    )
    #print("*"*25,"DB Results=>",results,"*"*25)
    response = generateContentUsingGenAI(results,query,SearchOutsideDB)
    return response
def getDataFromJSONFile(FileName,KeyName):
    with open(FileName, "r") as config_file:
        config_data = json.load(config_file)
    #print("*"*25,"JSON Data","*"*25)
    #print(config_data.get(KeyName))
    return config_data.get(KeyName)

def generateContentUsingGenAI(results,query,SearchOutsideDB):
    google_api_key = getDataFromJSONFile("config.JSON","google-ai-secret-key")
    genai.configure(api_key=google_api_key)
    prompt = getDataFromJSONFile("prompt_templates/rag_template_1.JSON","context")
    if SearchOutsideDB is True:
        prompt = getDataFromJSONFile("prompt_templates/rag_template_2.JSON", "context")

    prompt += f"""
    Context:
    {"".join(results["documents"][0])}
    User Question:
    {query}
    """
    # print("*" * 25, "Prompt Data", "*" * 25)
    # print(prompt)
    # print("*"*50)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    print("*" * 25, "Gen AI Response:", "*" * 25)
    print(response.text)
    print("*" * 50)
    return response.text

#QueryChromas("What is Linear regression?","TestCollection-2025-09-05-11-00-58",1,False )