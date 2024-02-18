from langchain_community.llms import HuggingFaceHub
# from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


load_dotenv()
repo_id = "huggingfaceh4/zephyr-7b-alpha"


def openai_generate_pet_name():
    llm = OpenAI(temperature=0.7)  # depracated
    ans = llm("prompt/question")
    return ans


def genertate_pet_name(animal_type):
    llm = HuggingFaceHub(repo_id=repo_id,
                         model_kwargs={"temperature": 0.5, "max_length": 50})

    # question = """I have a dog pet and I want a cool name for it.
    # Suggest my five cool names for my pet."""

    prompt_template_name = PromptTemplate(
        input_variables=["animal_type"],
        template="""
        <|system|>
        You are an AI assistant that follows instruction extremely well.
        Please be truthful and give direct answers
        </s>
        <|user|>
        I want a {animal_type} and I want a cool name for it. Suggest my five cool names for it.
        </s>
        <|assistant|>
        """
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    response = name_chain.invoke({"animal_type": animal_type})
    return(response)


def genertate_pet_name_2():
    question = """I have a dog pet and I want a cool name for it.
    Suggest my five cool names for my pet."""
    llm = HuggingFaceHub(repo_id=repo_id,
                         model_kwargs={"temperature": 0.7, "max_length": 50})
    name = llm.invoke(question)
    return(name)


if __name__ == "__main__":
    print(genertate_pet_name("cow"))
