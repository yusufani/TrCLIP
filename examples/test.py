# %%
# %%
from trclip.trclip import TrCLIP
from trclip.trclip import ModifiedTextEncoder


# %%
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained('yusufani/trclip-vitl14-e10')
tokenizer = transformers.AutoModelForSequenceClassification.from_pretrained('yusufani/trclip-vitl14-e10')
#%%

ModifiedTextEncoder().
#%%
import torch
encoding = tokenizer.batch_encode_plus(['selam'], return_tensors='pt', padding=True, truncation=True,add_special_tokens=True, verbose=True)
with torch.no_grad():
    text_embs = model(encoding["input_ids"].to('cpu'),
                                  encoding['attention_mask'].to('cpu'))



#%%
a = ModifiedTextEncoder('dbmdz/bert-base-turkish-cased' ,embeddingSize=768 )
#%%
import torch
a.load_state_dict(model.state_dict())
#%%
tokenizer = transformers.AutoTokenizer.from_pretrained('yusufani/trclip-vitl14-e10', use_auth_token=True)
#%%


config = transformers.AutoConfig.from_pretrained('dbmdz/bert-base-turkish-cased')
