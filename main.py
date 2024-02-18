from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
repo_id="gpt2"


def genertate_pet_name():
    question = """I have a dog pet and I want a cool name for it.
    Suggest my five cool names for my pet."""
    template = """Question: {question}
        Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 200})
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return(llm_chain.invoke(question))


def genertate_pet_name_2():
    question = """I have a dog pet and I want a cool name for it.
    Suggest my five cool names for my pet."""
    llm = HuggingFaceHub(repo_id=repo_id,
                         model_kwargs={"temperature": 0.5, "max_length": 200})
    name = llm.invoke(question)
    return(name)


if __name__ == "__main__":
    print(genertate_pet_name())