#%%
import openai,os
import timeit
from functools import partial
from dotenv import load_dotenv
env_path = '../../../.env'
load_dotenv(dotenv_path=env_path)

#%%
client = openai.Client(
    base_url=os.getenv("Netmind_BASEURL"), 
    api_key=os.getenv("Netmind_AIP_KEY")
    )

def get_llm_text_response(user_message,
                          client,
                          model_name="meta-llama-Meta-Llama-3.1-8B-Instruct",
                          max_tokens=4096):
    # Chat completion
    response = client.chat.completions.create(
        model=model_name, 
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
#%%
# Running the timeit 5 times
test_models = ["meta-llama-Meta-Llama-3.1-8B-Instruct",
               "meta-llama-Meta-Llama-3.1-70B-Instruct"]
for m in test_models:
    stmt = partial(get_llm_text_response,'please write a news article that is 1000 words long.',
                client,
                m,
                4096)
    execution_time = timeit.repeat(stmt=stmt, repeat=1, number=5)
    print("{} Time used: {:.2f}".format(m,sum(execution_time)))

#%%
user_messages = [
                'please write a news article that is 2000 words long.',
                'Create a simple game with python code',
                'Explain the difference between quantum physics and Newtonian physics.',
                'Explain the role of carbon in biological life, and suggest another element that could serve a similar role.',
                ]
for idx,u_m in enumerate(user_messages):
    print('Answer {}'.format(idx))
    print(get_llm_text_response(u_m,client,
                                "meta-llama-Meta-Llama-3.1-8B-Instruct",
                                4096))