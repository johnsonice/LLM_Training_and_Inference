#%%
import openai,os
import timeit
from dotenv import load_dotenv, find_dotenv
env_path = '../../../.env'
load_dotenv(dotenv_path=env_path)

#%%
client = openai.Client(
    base_url=os.getenv("Netmind_BASEURL"), 
    api_key=os.getenv("Netmind_AIP_KEY")
    )

def get_llm_text_response(user_message,client,max_tokens=4096):
    # Chat completion
    response = client.chat.completions.create(
        model="meta-llama-Meta-Llama-3.1-8B-Instruct", # "llama3-8b"  or "llama3-70b"
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

user_messages = [
                'please write a news article that is 2000 words long.',
                'Create a simple game with python code',
                'Explain the difference between quantum physics and Newtonian physics.',
                'Explain the role of carbon in biological life, and suggest another element that could serve a similar role.',
                 ]
for idx,u_m in enumerate(user_messages):
    print('Answer {}'.format(idx))
    print(get_llm_text_response(u_m,client,4096))
    
#%%

# # Running the timeit 10 times
# execution_time = timeit.repeat(stmt=get_llm_text_response, repeat=10, number=1)
# print("Time used: {:.2f}".format(sum(execution_time)))
