import os

import discord
from dotenv import load_dotenv
from neuralintents import GenericAssistant

chatbot = GenericAssistant('intents.json')
chatbot.train_model()
chatbot.save_model()

client = discord.Client()

load_dotenv()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$mişkek'):
        response = chatbot.request(message.content[8:])
        await message.channel.send(response)

client.run(os.getenv('TOKEN'))
