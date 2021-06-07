from PIL import Image, ImageDraw, ImageColor
import random

if __name__ == '__main__':
    colors = ['black', 'red', 'blue']
    for color in colors:
        for i in range(1, 100):
            im = Image.new('RGBA', (64, 64), 'white')
            start = (random.randint(0, 10), random.randint(0, 10))
            end = (random.randint(54, 64), random.randint(54, 64))
            draw = ImageDraw.Draw(im)
            draw.line(([start, end]), fill=color, width=3)
            name = color + '/' + color + str(i) + '.png'
            im.save(name)
            if i % 10 is 0:
                print(color, i)
