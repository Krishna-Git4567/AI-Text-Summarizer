from fastapi import FastAPI, Request
from pydantic import BaseModel # validat api request 
from transformers import T5ForConditionalGeneration , T5Tokenizer 
import torch 
import re 
from fastapi.templating import Jinja2Templates # UI 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# initialize our fast api app
app = FastAPI(title= "Web-Slinger Summarizer APP",description= "text summarizer using T5-small", version= "1.0" )

# MODEL AND TOKENIZER
model = T5ForConditionalGeneration.from_pretrained("../MODEL/save_summary_model")
tokenizer= T5Tokenizer.from_pretrained("../MODEL/save_summary_model")

# DEVICE 
if torch.backends.mps.is_available():
    device = torch.device("mps")
    
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# TEMPLETING 
templates = Jinja2Templates(directory= ".")

# input schema  for dialogue  = > string 

class DialogueInput(BaseModel):
    dialogue : str 


def clean_data(text):
    text = re.sub(r"\r\n",  " ", text) # extra lines 
    text = re.sub(r"\s+",   " ", text) # spaces 
    text = re.sub(r"<.*?>", " ", text) # html tage <p>
    text = text.strip().lower()
    return text

## summarizer F(X)
def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # data clean 

    # toknize
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensors = "pt"  # pytorch tensor 
        
    ).to(device)

    # genrateing summary = > token ids 
    model.to(device)
    target = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150 ,
        num_beams = 4 ,  # 4 output we have to choice one 
        early_stopping = True
    )

    # token ids convert to summary => decoding 

    summary = tokenizer.decode(target[0], skip_special_tokens= True) # EOS , SEP AND ALL 
    return summary 

## APPI ENDPOINTS 

@app.post("/summarize")
async def summarize(dialogue_input :DialogueInput ):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class = HTMLResponse)
async def home(request: Request ):
    return templates.TemplateResponse(
    request=request,
    name="index.html"
)
 
