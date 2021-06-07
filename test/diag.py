from PIL import Image, ImageDraw, ImageColor
import random

if __name__ == '__main__':
    colors = ['tlbr', 'bltr']
    for i in range(1, 100):
        im = Image.new('RGBA', (64, 64), 'white')
        start = (random.randint(0, 10), random.randint(0, 10))
        end = (random.randint(54, 64), random.randint(54, 64))
        draw = ImageDraw.Draw(im)
        draw.line(([start, end]), fill='black', width=3)
        name = 'tlbr' + '/' + 'tlbr' + str(i) + '.png'
        im.save(name)
    for i in range(1, 100):
        im = Image.new('RGBA', (64, 64), 'white')
        start = (random.randint(0, 10), random.randint(54, 64))
        end = (random.randint(54, 64), random.randint(0, 10))
        draw = ImageDraw.Draw(im)
        draw.line(([start, end]), fill='black', width=3)
        name = 'bltr' + '/' + 'bltr' + str(i) + '.png'
        im.save(name)
