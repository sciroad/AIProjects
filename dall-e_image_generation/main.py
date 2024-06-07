import openai 


openai.api_key = open('API_KEY','r').read().strip()

response= openai.Image.create(
    prompt="A painting of a bird",
    model="image-encoder-v1",
    n=1,
    size='1024x1024',
)


image_url = response['data'][0]['url']

print(image_url)
