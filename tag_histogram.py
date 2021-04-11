import numpy as np
import os
from os import listdir, mkdir, path
from matplotlib import pyplot as plt
from collections import Counter
import shutil


def get_or_make_subfolder(tag):
    sorted_path = os.environ['CGAN_SORTED']
    dir = sorted_path + f'/{tag}'
    if not path.isdir(dir):
        try:
            mkdir(dir)
        except:
            pass
    return dir


def sort_pictures():
    all_tags = []
    dir = os.environ['CGAN_IMAGE_PATH']
    artists = listdir(dir)
    for artist in artists:
        print(f'Currently at {artist}')
        curr_dir = dir + '\\' + artist
        songs = listdir(curr_dir)
        # -4 to cut off the .png
        try:
            artist_tags = songs[0].split(';')[-1][:-4].split('#')
        except:
            print(f'Error with {artist}')
        for artist_tag in artist_tags:
            tag_dir = get_or_make_subfolder(artist_tag)
            for song in songs:
                song_path = dir + f'/{artist}/{song}'
                if not os.path.isfile(tag_dir + f'/{song}'):
                    shutil.copy(song_path, tag_dir)

        all_tags += artist_tags

    counter = Counter(all_tags)


def get_dict() -> dict:
    tag_path = os.environ['CGAN_SORTED']
    tag_list = os.listdir(tag_path)
    tag_dict = {}
    for tag in tag_list:
        curr_path = tag_path + f'/{tag}'
        song_list = listdir(curr_path)
        tag_dict[tag] = len(song_list)
    return tag_dict


if __name__ == '__main__':
    a = get_dict()
    print(max(a, key=lambda k: a[k]))
