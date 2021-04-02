import musicbrainzngs as brainz
import logging
import io
import PIL.Image as Image
from numba import njit
import string
from os import mkdir, path, getcwd, environ


def get_subdirectory(artist):
    name = dont_fuck_up_path(artist['name'])
    image_path = environ['CGAN_IMAGE_PATH'] if 'CGAN_IMAGE_PATH' in environ.keys() else "./images"
    dir = image_path + f'/{name}'
    if not path.isdir(dir):
        try:
            mkdir(dir)
        except OSError:
            try:
                alias_name = artist['alias-list'][0]['alias']
                dir = image_path + f'/{alias_name}'
            except KeyError:
                shorter_name = name[:50] + '...'
                dir = image_path + f'/{shorter_name}'
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
    start_artist = 0
    release_start = 0
    artist_limitsize = 100
    release_limitsize = artist_limitsize
    letters = ['w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n']

    for letter in letters:
        query: dict = brainz.search_artists(letter, limit=artist_limitsize)
        artist_count = query['artist-count']
        artist_offset = start_artist // artist_limitsize * artist_limitsize
        while artist_offset < artist_count:
            query: dict = brainz.search_artists(letter, limit=artist_limitsize, offset=artist_offset)
            for artist_index, artist in enumerate(query['artist-list'][start_artist % artist_limitsize:]):
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
                # length of all releases from this artist
                album_query = brainz.browse_releases(artist=artist["id"])
                album_query_len = album_query['release-count']
                # initialize temp variables
                saved_releases = []
                release_offset = release_start//release_limitsize * release_limitsize
                while release_offset < album_query_len:
                    for release_index, release in enumerate(brainz.browse_releases(
                            artist=artist['id'],
                            offset=release_offset + release_start,
                            limit=release_limitsize)['release-list']):
                        current_artist = artist_offset + artist_index + start_artist % artist_limitsize
                        current_release = release_offset + release_index + release_start % release_limitsize
                        print(
                            f"Letter: {letter}, Artist: {artist['name']} \n"
                            f"ArtistIndex:{current_artist} ({(current_artist / artist_count * 100):.1f}%), "
                            f"ReleaseIndex:{current_release} ({(current_release / album_query_len * 100):.1f}%)")
                        if release['title'] in saved_releases:
                            print("SKIIIIIP : RELEASE ALREADY SAVED")
                            continue
                        else:
                            saved_releases.append(release['title'])
                        if release['cover-art-archive']['artwork'].lower() == 'false':
                            print("SKIIIIIP : RELEASE DOES NOT HAVE ARTWORK")
                            continue
                        subdirectory = get_subdirectory(artist)
                        try:
                            filename = f"{release['date'].split('-')[0]};{release['title']};{genre}.png"
                            correct_path = f"{subdirectory}/{dont_fuck_up_path(filename)}"
                            if path.exists(correct_path):
                                print("SKIIIIIP : RELEASE WAS ALREADY SAVED BEFORE!!")
                                continue
                        except KeyError:
                            print("SKIIIIIP : RELEASE DOES NOT HAVE DATE OR TITLE")
                            continue
                        # donwload image
                        try:
                            art = brainz.get_image_front(release['id'], size=250)
                            image = Image.open(io.BytesIO(art))
                        except brainz.ResponseError:
                            print("SKIIIIIP : ARTWORK IS NOT AVAIABLE IN 250")
                            continue
                        except brainz.NetworkError:
                            print("SKIIIIIP : NETWORK ERROR.")
                            continue
                        try:
                            image.save(correct_path)
                            print("FILE SAVED")
                        except:
                            print("UNKNOWN ERROR OCCURED!!!")
                            continue
                    release_offset += release_limitsize
                    release_start = 0
            artist_offset += artist_limitsize
            start_artist = 0
