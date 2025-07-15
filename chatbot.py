from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_response(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_return_sequences=1
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def chat():
    print("🤖 Advanced ChatBot (type 'exit' to quit)")
    context = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatBot: Goodbye!")
            break
        prompt = context + f"You: {user_input}\nBot:"
        response = generate_response(prompt)
        bot_reply = response.replace(prompt, "").split('\n')[0].strip()
        print(f"ChatBot: {bot_reply}")
        context += f"You: {user_input}\nBot: {bot_reply}\n"

if __name__ == "__main__":
    chat()
