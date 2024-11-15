import discord
from discord.ext import commands
from keras.api.models import load_model
from PIL import Image
import requests
import io
from discord import Intents, Activity, ActivityType, Embed
from keras.api.layers import BatchNormalization, DepthwiseConv2D
import numpy as np
import h5py
import tensorflow as tf
import asyncio
import os

TOKEN = os.getenv('DISCORD_TOKEN')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs found: {len(gpus)}. Using GPU for inference.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU for inference.")

# Fix for TensorFlow model issues
f = h5py.File("C:/Users/Yerov/PycharmProjects/ImgAuthy/keras_model.h5", mode="r+")
model_config_string = f.attrs.get("model_config")
if model_config_string.find('"groups": 1,') != -1:
    model_config_string = model_config_string.replace('"groups": 1,', '')
    f.attrs.modify('model_config', model_config_string)
    f.flush()
f.close()

model = load_model("C:/Users/Yerov/PycharmProjects/ImgAuthy/keras_model.h5", custom_objects={'Normalization': BatchNormalization, 'DepthwiseConv2D': DepthwiseConv2D})

# Compile the model (necessary for certain Keras versions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

CORRECT_CLASS_LABEL = 1

intents = Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents, activity=Activity(type=ActivityType.playing, name="My Game"))

user_avatars = {}
# Log channel ID
LOG_CHANNEL_ID = 123456789012345678

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}!')
    bot.loop.create_task(auto_reverify())

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

@bot.event
async def on_member_update(before, after):
    if before.avatar != after.avatar:
        await on_profile_change(after)

@bot.command()
async def verify(ctx):
    await verify_user(ctx.author)

@bot.command()
@commands.has_permissions(administrator=True)
async def set_verification_role(ctx, *, role_name: str):
    global verification_role_name
    verification_role_name = role_name
    await ctx.send(f"Verification role updated to: {verification_role_name}")

@bot.command()
async def verification_stats(ctx):
    verified_count = len([user for user in user_avatars if user_avatars[user]])
    await ctx.send(f"Currently verified users: {verified_count}")

@bot.command()
@commands.has_permissions(administrator=True)
async def check_avatar(ctx, user: discord.User):
    current_avatar_url = user.avatar.url
    embed = Embed(title=f"Avatar of {user}", color=0x00FFFF)
    embed.set_image(url=current_avatar_url)
    await ctx.send(embed=embed)

async def log_message(content: str):
    channel = bot.get_channel(LOG_CHANNEL_ID)
    if channel:
        await channel.send(content)

async def on_profile_change(user):
    current_avatar_url = user.avatar.url
    previous_avatar_url = user_avatars.get(user.id)

    if previous_avatar_url is None or previous_avatar_url != current_avatar_url:
        user_avatars[user.id] = current_avatar_url
        await verify_user(user)

async def verify_user(user):
    embed = Embed(title="Verification Result", color=0xFFFF00)
    embed.add_field(name="Status", value="⏳ Verifying...", inline=False)
    embed.add_field(name="Details", value="Please wait while we verify your profile picture.", inline=False)

    channel = user.dm_channel if user.dm_channel else await user.create_dm()
    status_message = await channel.send(embed=embed)

    current_avatar_url = user.avatar.url
    try:
        response = requests.get(current_avatar_url)
        response.raise_for_status()
    except requests.RequestException as e:
        await status_message.edit(embed=Embed(title="Verification Failed", description=f"❌ Could not retrieve profile picture: {e}", color=0xFF0000))
        return

    image_data = response.content
    image = Image.open(io.BytesIO(image_data)).convert('RGB').resize((224, 224))
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)

    if index == CORRECT_CLASS_LABEL:
        embed = Embed(title="Verification Result", color=0x00FF00)  # Green for success
        embed.add_field(name="Status", value="✅ Verified", inline=False)
        embed.add_field(name="Details", value=f"{user.mention}, access granted based on your profile picture!", inline=False)

        role = discord.utils.get(user.guild.roles, name="Verified")
        if role:
            await user.add_roles(role)
            user_avatars[user.id] = current_avatar_url
        await log_message(f"User {user} has been verified.")
    else:
        embed = Embed(title="Verification Result", color=0xFF0000)
        embed.add_field(name="Status", value="❌ Access Denied", inline=False)
        embed.add_field(name="Details", value=f"{user.mention}, your profile picture did not match the key.", inline=False)

        role = discord.utils.get(user.guild.roles, name="Verified")
        if role and role in user.roles:
            await user.remove_roles(role)
        await log_message(f"User {user} failed verification.")

    await status_message.edit(embed=embed)

async def auto_reverify():
    await bot.wait_until_ready()
    while not bot.is_closed():
        for user_id in user_avatars.keys():
            user = bot.get_user(user_id)
            if user:
                await verify_user(user)
        await asyncio.sleep(2592000)  # Reverify every 30 days

bot.run(TOKEN)
