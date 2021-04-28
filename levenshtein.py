from Levenshtein import distance
import os
import sys
import shutil
from os import listdir, path


def levenshtein_distance():
    path = os.environ['CGAN_SORTED']
    genres = os.listdir(path)
    for index1, genre1 in enumerate(genres):
        for index2, genre2 in enumerate(genres[index1 + 1:]):
            dist = distance(genre1, genre2)
            if dist < 2:
                print(genre1, genre2)


def user_input():
    text = input('prompt')
    print(text)


def combine_folders(path1: str, path2: str, first: bool):
    if path1 == path2:
        print('Same directory?')
        return
    root_files = listdir(path1) if not first else listdir(path2)
    root_folder = path1 if not first else path2
    target_folder = path1 if first else path2
    for file in root_files:
        file_path = root_folder + f'/{file}'
        if not path.isfile(target_folder + f'/{file}'):
            shutil.copy(file_path, target_folder)
        else:
            print('Already copied!')
        os.remove(file_path)
    if len(listdir(root_folder)) == 0:
        os.rmdir(root_folder)
    else:
        print('Error removing folder ' + root_folder)


def compare_folders():
    path = os.environ['CGAN_SORTED']
    genrelist = listdir(path)
    i1, i2 = 160, 1
    while i1 < len(listdir(path)):
        progress = float(i1 * 100 / len(listdir(path)))
        print(f'{progress:.2f} % \t {listdir(path)[i1]}')
        while i2 < len(listdir(path)):
            genre1 = listdir(path)[i1]
            l1 = len(listdir(path + f'/{genre1}'))
            genre2 = listdir(path)[i2]
            l2 = len(listdir(path + f'/{genre2}'))
            dist = distance(genre1, genre2)
            if dist < 3:
                print(f'1: {genre1} \t 2: {genre2} \t 3: Skip \n Length: {l1} \t Length: {l2}')
                first = input('')
                if first == '1':
                    combine_folders(path1=path + f'/{genre1}',
                                    path2=path + f'/{genre2}',
                                    first=True)
                    continue
                elif first == '2':
                    combine_folders(path1=path + f'/{genre1}',
                                    path2=path + f'/{genre2}',
                                    first=False)
                    continue
                elif first == '3':
                    i2 += 1
                    continue
                else:
                    print('Error, wrong Input!')
            else:
                i2 += 1
        i1 += 1
        i2 = i1 + 1


if __name__ == '__main__':
    compare_folders()
