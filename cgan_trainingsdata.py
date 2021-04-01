import musicbrainzngs as brainz
import logging
import io
import PIL.Image as Image
from numba import njit
import string
from os import mkdir, path, getcwd


def get_subdirectory(sd):
    name = sd['name']
    dir = path.join(getcwd(), f'images/{name}')
    if not path.isdir(dir):
        mkdir(dir)
    return dir


def dont_fuck_up_path(path: str) -> str:
    invalid_characters = ['<', '>', ':', r'"', r'/', '\\', r'|', '?', '*']
    for char in invalid_characters:
        path = path.replace(char, '')
    return path


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # brainz.set_hostname(\"coverartarchive.org\")
    brainz.set_useragent("cgan_trainings_data_grabber", "0.0.1", "virtualnonsense@mailbox.org")

    '''TODO:
    search every letter
    iterate through artists and get genre
    look through records
    see if album has cover
    get album with date at size 250
    '''
    alphabet_list = list(string.ascii_lowercase)
    start_artist = 256
    release_start = 19
    limitsize = 100
    letter = 'a'

    query: dict = brainz.search_artists(letter, limit=limitsize)
    artist_count = query['artist-count']
    query_offset = start_artist // limitsize * limitsize

    while query_offset < artist_count:
        query: dict = brainz.search_artists(letter, limit=limitsize, offset=query_offset)
        for artist_index, artist in enumerate(query['artist-list'][start_artist % limitsize:]):
            # get genres for artist (not more than 3)
            if not str(artist['name']).lower().startswith(letter):
                print(f"SKIIIIIP : ARTIST DOES NOT START WITH {letter}")
                continue
            try:
                newlist = sorted(artist['tag-list'], key=lambda k: int(k['count']), reverse=True)
                genre_list = [k['name'] for k in newlist]
                genre = '#'.join(genre_list if len(genre_list) < 4 else genre_list[:3])
            except KeyError:
                print("SKIIIIIP : ARTIST HAS NO GENRE TAGS")
                continue

            # get albums
            album_query = brainz.browse_releases(artist=artist["id"])
            saved_releases = []
            for release_index, release in enumerate(album_query['release-list'][release_start:]):
                print(f"ArtistIndex:{query_offset + artist_index + start_artist % limitsize}, "
                      f"ReleaseIndex:{release_index + release_start}")
                if release['title'] in saved_releases:
                    print("SKIIIIIP : RELEASE ALREADY SAVED")
                    continue
                if release['cover-art-archive']['artwork'].lower() == 'false':
                    print("SKIIIIIP : RELEASE DOES NOT HAVE ARTWORK")
                    continue
                saved_releases.append(release['title'])
                try:
                    art = brainz.get_image_front(release['id'], size=250)
                except brainz.ResponseError:
                    print("SKIIIIIP : ARTWORK IS NOT AVAIABLE IN 250")
                    continue
                image = Image.open(io.BytesIO(art))
                subdirectory = get_subdirectory(artist)
                try:
                    filename = f"{release['date'].split('-')[0]};{release['title']};{genre}.png"
                    correct_path = f"{subdirectory}/{dont_fuck_up_path(filename)}"
                    image.save(correct_path)
                except KeyError:
                    print("SKIIIIIP : RELEASE DOES NOT HAVE DATE OR TITLE")
            release_start = 0
        query_offset += limitsize
        start_artist = 0
