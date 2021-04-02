import musicbrainzngs as brainz
import logging
import io
import PIL.Image as Image
from numba import njit
import string
from os import mkdir, path, getcwd, environ


def get_subdirectory(curr_artist):
    name = dont_fuck_up_path(curr_artist['name'])
    # get path to image folder, in os env variables or subfolder of current
    image_path = environ['CGAN_IMAGE_PATH'] if 'CGAN_IMAGE_PATH' in environ.keys() else "./images"
    dir = image_path + f'/{name}'
    # check if already created
    if not path.isdir(dir):
        try:
            mkdir(dir)
        # exception when dir name is to long -> check if artist has abbrv.
        except OSError:
            try:
                alias_name = curr_artist['alias-list'][0]['alias']
                dir = image_path + f'/{alias_name}'
            # just shorten the name and check if dir with shortend name exists
            except KeyError:
                shorter_name = name[:50] + '...'
                dir = image_path + f'/{shorter_name}'
            if not path.isdir(dir):
                mkdir(dir)
    return dir


# remove all invalid characters where windows thinks they're stupid
def dont_fuck_up_path(wrong_path: str) -> str:
    invalid_characters = ['<', '>', ':', r'"', r'/', '\\', r'|', '?', '*']
    for char in invalid_characters:
        corr_path = wrong_path.replace(char, '')
    return corr_path


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
    # set starting indices
    start_artist = 4673
    release_start = 642
    artist_limitsize = 100
    release_limitsize = artist_limitsize
    letters = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']

    for letter in letters:
        # get a dict with all artists when searching for one letter
        query: dict = brainz.search_artists(letter, limit=artist_limitsize)
        artist_count = query['artist-count']
        artist_offset = start_artist // artist_limitsize * artist_limitsize
        # brainz.search can only show 100 max, so go through loop multiple times
        while artist_offset < artist_count:
            # get dict with artists of letter at offset point
            query: dict = brainz.search_artists(letter, limit=artist_limitsize, offset=artist_offset)
            for artist_index, artist in enumerate(query['artist-list'][start_artist % artist_limitsize:]):
                # skip all artists that do not start with current letter
                if not str(artist['name']).lower().startswith(letter):
                    print(f"SKIIIIIP : ARTIST DOES NOT START WITH {letter}")
                    continue
                # get genres for artist (not more than 3)
                try:
                    # sort tags by count
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
                release_offset = release_start // release_limitsize * release_limitsize
                # same as above, brainz.search has limit of 100
                while release_offset < album_query_len:
                    for release_index, release in enumerate(brainz.browse_releases(
                            artist=artist['id'],
                            offset=release_offset + release_start,
                            limit=release_limitsize)['release-list']):
                        # calculate current position for progress debug
                        current_artist = artist_offset + artist_index + start_artist % artist_limitsize
                        current_release = release_offset + release_index + release_start % release_limitsize
                        print(
                            f"Letter: {letter}, Artist: {artist['name']} \n"
                            f"ArtistIndex:{current_artist} ({(current_artist / artist_count * 100):.1f}%), "
                            f"ReleaseIndex:{current_release} ({(current_release / album_query_len * 100):.1f}%)")
                        # skip if already saved
                        if release['title'] in saved_releases:
                            print("SKIIIIIP : RELEASE ALREADY SAVED")
                            continue
                        else:
                            saved_releases.append(release['title'])
                        # check if release has coverart
                        if release['cover-art-archive']['artwork'].lower() == 'false':
                            print("SKIIIIIP : RELEASE DOES NOT HAVE ARTWORK")
                            continue
                        # get file location
                        subdirectory = get_subdirectory(artist)
                        try:
                            filename = f"{release['date'].split('-')[0]};{release['title']};{genre}.png"
                            correct_path = f"{subdirectory}/{dont_fuck_up_path(filename)}"
                            # check if file with filename exists -> was already saved before
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
                    # reset variables
                    release_offset += release_limitsize
                    release_start = 0
            artist_offset += artist_limitsize
            start_artist = 0
