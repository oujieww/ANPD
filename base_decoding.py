import pickle
import torch
import torch.nn as nn
import time
from transformers import top_k_top_p_filtering
from collections import defaultdict, deque

def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids

def base_generate(model, tokenizer, input_ids,  bi_model=None, query=None, max_new_tokens=10, 
                  do_sample=False, top_k=0, top_p=0.85, temperature=0.2,
                  early_stop=False):

    current_input_ids = input_ids
    # print(input_ids)
    generate_ids = torch.empty([input_ids.size(0), max_new_tokens], dtype=torch.long, device=model.device)
    past_key_values = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            output = model(input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            logits = output['logits'][:,-1:]
            output_ids = sample(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            past_key_values = output['past_key_values']

            if early_stop and current_input_ids.item() == tokenizer.eos_token_id:
                break

    step = min(step+1, max_new_tokens)
    generate_ids = generate_ids[:, :step]
                
    return {
        'generate_ids': generate_ids,
    }
