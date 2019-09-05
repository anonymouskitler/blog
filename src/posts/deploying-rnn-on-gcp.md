---
title: Deploy simple language model on serverless stack using Google cloud functions
date: '2019-08-29'
tags:
  - rnn
  - language-model
  - nlp
  - deep-learning
  - serverless
  - blog
  - pytorch
---
In this article we will deploy a simple random text generator(using RNN) built in pytorch on serverless architectures. The main motivation was to get comfortable with productionising simple deep learning models & deploy them at minimal cost using serverless.
- - -
1. Pretrain a simple RNN char model
2. Export the model weights using model.save()
3. Export the model artefacts (for language model it's vocabulary & vocab mappings)
4. Create an S3 bucket
5. Upload the model & artefacts to S3 and give sufficient permissions (here I gave unrestricted access to mine).
6. Create a google cloud function
7. Edit code to load model, predict and send them as API response in the cloud function. 
- - -

### Pretrain a simple RNN char model
To achieve our objective without complications in our deep learning language model, we will use a simplistic dataset. Which is [Nietzsche's articles](https://s3.amazonaws.com/text-datasets/nietzsche.txt). The following RNN model is heavily inspired from Jeremy Howard's fast.ai lecture on RNNs. 

The model itself is quite simplistic. It is a char RNN model - we tokenize the complete text corpus on characters instead of words & sentences. I have explained in another article how to prepare & load data to build such RNN models from scratch.

```python
class Char3Model(nn.Module):
    def __init__(self, vocab_size, n_fac, n_hidden):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac, n_hidden)
        self.l_h = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)

    def forward(self, c1, c2, c3):
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))

        h = torch.zeros(in1.size(),requires_grad=True)
        h = F.tanh(self.l_h(h+in1))
        h = F.tanh(self.l_h(h+in2))
        h = F.tanh(self.l_h(h+in3))

        return F.log_softmax(self.l_out(h))
```
The model takes 3 characters as input and outputs a single character. We will write helper functions to generate text by feeding the ouput again as input to the model. 

- - -

### Export the model weights

[Pytorch](https://pytorch.org) provides an easy interface to save and load trained models. To know more about pytorch interface to load and save models, check out their awesome [tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

Let's save out model weights:

```python
torch.save(model, PATH_TO_FILE + '/' + MODEL_FILE_NAME)
```
`PATH_TO_FILE` being the file path where you want to export the model.

- - -

### Export the model artefacts
In case our CharRNN model, it's just vocabulary which is our characters found in the corpus & the character to token mapping which is basically a hashmap of the character and it's respective character code. They look like:

Characters:
```python
chars=['\x00','\n',' ','!','"',"'",'(',')',',','-','.','0','1','2','3','4','5','6','7','8','9',':',';','=','?','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[',']','_','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','Æ','ä','æ','é','ë']
```

Character mapping:
```python
char_indices={'\x00':0,'\n':1,' ':2,'!':3,'"':4,"'":5,'(':6,')':7,',':8,'-':9,'.':10,'0':11,'1':12,'2':13,'3':14,'4':15,'5':16,'6':17,'7':18,'8':19,'9':20,':':21,';':22,'=':23,'?':24,'A':25,'B':26,'C':27,'D':28,'E':29,'F':30,'G':31,'H':32,'I':33,'J':34,'K':35,'L':36,'M':37,'N':38,'O':39,'P':40,'Q':41,'R':42,'S':43,'T':44,'U':45,'V':46,'W':47,'X':48,'Y':49,'Z':50,'[':51,']':52,'_':53,'a':54,'b':55,'c':56,'d':57,'e':58,'f':59,'g':60,'h':61,'i':62,'j':63,'k':64,'l':65,'m':66,'n':67,'o':68,'p':69,'q':70,'r':71,'s':72,'t':73,'u':74,'v':75,'w':76,'x':77,'y':78,'z':79,'Æ':80,'ä':81,'æ':82,'é':83,'ë':84}
```

Since the vocab (characters) & token mappings are single line, let's just copy & paste them in the cloud function. Alternatively we can export in pickle format, upload them to S3 & read them dynamically in our cloud function.

### Create an S3 bucket

AWS S3 is a cloud data storage service. We store data in *buckets* here (to grossly simplify it assume them to be akin to folders in your desktop). Alternatively you can use Google or Azure storage services.

If you don't have an Amazon AWS account, I'd recommend creating one now. I think their free tier is pretty generous. If you already do, let's go ahead and create a S3 bucket to upload our model artefacts.

Before we proceed install [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) on our machine and [configure it](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

After we have configured our aws cli, let's create an s3 bucket from our commandline. Alternatively we can create using GUI on aws console. 

```bash
aws s3 mb s3://REPLACE_WITH_YOUR_BUCKET_NAME
```

### Upload the model & artefacts to S3 and give sufficient permissions

Install aws python SDK.
```shell
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#installation
```

Let's upload our model to s3 using the aws python sdk. 

```python
import boto3
s3 = boto3.resource('s3')
s3.meta.client.upload_file('PATH_TO_FILE', 'REPLACE_WITH_YOUR_BUCKET_NAME', 'MODEL_FILE_NAME')
```
The file should be uploaded to s3 after the above step. 
Go to S3 bucket permission settings tab in the aws console and set *Block all public access* to *off*. 

Go to the bucket properties and copy the link to the model. We will be using that link to download the model.

### Create a google cloud function
Log on to [GCP console](https://console.cloud.google.com) and choose Cloud functions. Even GCP offers a generous free tier with $300 credits for one year. 

Click on to create function and a form appears. Fill the form with function name (CharRNN), set the memory allocated to 512MB, set runtime as Python 3.7. Leave the rest as it is. 

Click on create function. Copy the function url and open it in new tab. You should see 'Hello world' response.

- - -

### Edit code to load model, predict and send them as API response in the cloud function

Now we need to fill in the requirements.txt & main.py file using the on screen code editor. Click on the edit function button and let's start filling in those files one after the other.

In requirements.txt file:
```shell
# Function dependencies, for example:
# package>=version
requests==2.18.4
https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whldill>=0.2.8
numpy>=1.15.0 
```

In main.py file:

```python
import os
import requests

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# Our vocab
chars=['\x00','\n',' ','!','"',"'",'(',')',',','-','.','0','1','2','3','4','5','6','7','8','9',':',';','=','?','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[',']','_','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','Æ','ä','æ','é','ë']

# Vocab and code mappings
char_indices={'\x00':0,'\n':1,' ':2,'!':3,'"':4,"'":5,'(':6,')':7,',':8,'-':9,'.':10,'0':11,'1':12,'2':13,'3':14,'4':15,'5':16,'6':17,'7':18,'8':19,'9':20,':':21,';':22,'=':23,'?':24,'A':25,'B':26,'C':27,'D':28,'E':29,'F':30,'G':31,'H':32,'I':33,'J':34,'K':35,'L':36,'M':37,'N':38,'O':39,'P':40,'Q':41,'R':42,'S':43,'T':44,'U':45,'V':46,'W':47,'X':48,'Y':49,'Z':50,'[':51,']':52,'_':53,'a':54,'b':55,'c':56,'d':57,'e':58,'f':59,'g':60,'h':61,'i':62,'j':63,'k':64,'l':65,'m':66,'n':67,'o':68,'p':69,'q':70,'r':71,'s':72,'t':73,'u':74,'v':75,'w':76,'x':77,'y':78,'z':79,'Æ':80,'ä':81,'æ':82,'é':83,'ë':84}

# Get prediction for the given 3 characters
def get_next(inp, model):
    # Map the input chars to respective codes and load them in to a tensor
    idxs = torch.tensor(np.array([[char_indices[c]] for c in inp]), requires_grad=False)
    # Predict the model output by passing in the input
    p = model(*idxs)
    # These will return an array of probabilites, do argmax to get the index of char with max probability
    i = np.argmax(p.detach().numpy())
    # Return the output character
    return chars[i]

# Recursively call get_next function to get 'n' length text predictions 
def get_next_n(inp, n, model):
    print('predicting chars for inp ', inp)
    print('calling predict for next n ', n)
    res = inp
    for i in range(n):
        c = get_next(inp, model)
        # Append the result
        res += c
        # Update the input params
        inp = inp[1:]+c
    return res


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    MODEL_URL = 'https://xxx.xxxx.amazonaws.com/MODEL_FILE_NAME'

    r = requests.get(MODEL_URL)
    # Download & save it in tmp folder
    file = open("/tmp/char_3_model", "wb")
    file.write(r.content)
    file.close()
    print('got model')

    
    # The model arguments
    vocab_size = 85
    n_fac = 42 # embedding size
    n_hidden = 256 # no of hidden layers

    class Char3Model(nn.Module):
        def __init__(self, vocab_size, n_fac, n_hidden):
            super().__init__()
            self.e = nn.Embedding(vocab_size, n_fac)
            self.l_in = nn.Linear(n_fac, n_hidden)
            self.l_h = nn.Linear(n_hidden, n_hidden)
            self.l_out = nn.Linear(n_hidden, vocab_size)

        def forward(self, c1, c2, c3):
            in1 = F.relu(self.l_in(self.e(c1)))
            in2 = F.relu(self.l_in(self.e(c2)))
            in3 = F.relu(self.l_in(self.e(c3)))

            h = torch.zeros(in1.size(),requires_grad=True)
            h = F.tanh(self.l_h(h+in1))
            h = F.tanh(self.l_h(h+in2))
            h = F.tanh(self.l_h(h+in3))

            return F.log_softmax(self.l_out(h))

    # Initialize the model with model arguments
    model = Char3Model(vocab_size, n_fac, n_hidden)
    # State dict requires model object. Load the downloaded pytorch model from tmp folder
    model.load_state_dict(torch.load('/tmp/char_3_model'))

    print('loaded model')

    if request.method == 'GET':
        return " Welcome to nlp text generator"
    # We are using a POST request
    if request.method == 'POST':
        data = request.get_json()
        print("Request data is", data)
        x_data = data['text']
        text = get_next_n(x_data, 40, model)
    return text

```
After making the above changes, click on deploy button. 

We should be able to cURL from terminal and see the output:

```bash
curl --request POST \
  --url https://us-central1-centering-helix-251102.cloudfunctions.net/text-generator \
  --header 'content-type: application/json' \
  --data '{
	"text": "the"
}'
```

![Output of the above cURL command](/images/cURL-output-RNN.png "Text generated by our deployed model")

## Wrapping up

Hopefully this article was able to shed some light on deploying RNN models on Google cloud functions. If it did, [drop me a tweet](https://twitter.com/chandureddivari)!

