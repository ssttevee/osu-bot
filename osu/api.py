from __future__ import print_function

import io
import json
import os

import requests

from .beatmap import BeatmapSet


# Language Codes
Language_Any = "0"
Language_Other = "1"
Language_English = "2"
Language_Japanese = "3"
Language_Chinese = "4"
Language_Instrumental = "5"
Language_Korean = "6"
Language_French = "7"
Language_German = "8"
Language_Swedish = "9"
Language_Spanish = "10"
Language_Italian = "11"


# Genre Codes
Genre_Unspecified = "1"
Genre_VideoGame = "2"
Genre_Anime = "3"
Genre_Rock = "4"
Genre_Pop = "5"
Genre_Other = "6"
Genre_Novelty = "7"
Genre_HipHop = "9"
Genre_Electronic = "10"


# Status Codes
RankStatus_RankedAndApproved = "0"
RankStatus_Approved = "1"
RankStatus_Favourites = "2"
RankStatus_Qualified = "3"
RankStatus_Pending = "4"
RankStatus_Graveyard = "5"
RankStatus_MyMaps = "6"
RankStatus_Any = "7"
RankStatus_Loved = "8"


# Sorting
Sort_Title_Desc = "title_desc"
Sort_Title_Asc = "title_asc"
Sort_Artist_Desc = "artist_desc"
Sort_Artist_Asc = "artist_asc"
Sort_Difficulty_Desc = "difficulty_desc"
Sort_Difficulty_Asc = "difficulty_asc"
Sort_Ranked_Desc = "ranked_desc"
Sort_Ranked_Asc = "ranked_asc"
Sort_Rating_Desc = "rating_desc"
Sort_Rating_Asc = "rating_asc"
Sort_Plays_Desc = "plays_desc"
Sort_Plays_Asc = "plays_asc"


class OsuSearchResultIterator:
    def __init__(self, search_fn, limit):
        self.__search = search_fn
        self.__limit = limit
    
    def __iter__(self):
        self.__page = 0
        self.__pos = -1
        self.__total = 0
        self.__results = None
        return self
    
    def __next_results(self):
        self.__total += self.__pos + 1
        
        self.__page += 1
        self.__pos = 0
        
        self.__results = self.__search(self.__page)
    
    def __next__(self):
        self.__pos += 1
        
        if self.__limit <= self.__total + self.__pos:
            raise StopIteration
        
        if self.__page < 1 or self.__pos >= len(self.__results):
            self.__next_results()
        
        if len(self.__results) < 1:
            raise StopIteration
        
        return self.__results[self.__pos]


class OsuAPIClient:
    __SESSION_COOKIE__ = "osu_session"
    __CACHE_DIR__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".osu_cache")
    
    def __init__(self, username=None, password=None):
        self.__session = None
        
        if username is not None and password is not None:
            self.__login(username, password)
    
    def __login(self, username, password):
        r = requests.post("https://osu.ppy.sh/session", params={
            "username": username,
            "password": password,
        })
        
        resp = json.loads(r.text)
        
        if r.status_code != 200:
            try:
                raise RuntimeError(resp["error"])
            except KeyError:
                raise RuntimeError("Got non 200 error, but no error: " + r.text)
        
        self.__session = r.cookies[OsuAPIClient.__SESSION_COOKIE__]
    
    def logout(self):
        self.__session = None
    
    def search(self, language=None, sort=None, mode=None, rank_status=None, genre=None, limit=None):
        params = {}
        if language is not None:
            params["l"] = language
        
        if sort is not None:
            params["sort"] = sort
        
        if mode is not None:
            params["m"] = mode
        
        if rank_status is not None:
            params["s"] = rank_status
        
        if genre is not None:
            params["g"] = genre
        
        def search_fn(page):
            r = requests.get("https://osu.ppy.sh/beatmapsets/search", params={
                "page": str(page),
                **params,
            })
            
            return r.json()
        
        return OsuSearchResultIterator(search_fn, limit)
    
    def download_beatmap(self, beatmap_id, cache=True):
        cache_path = "%s/%s.osz" % (OsuAPIClient.__CACHE_DIR__, str(beatmap_id))
        if os.path.exists(cache_path):
            return BeatmapSet(open(cache_path, "rb"))
        
        if self.__session is None:
            raise RuntimeError("Must be logged in to download a beatmap")
        
        r = requests.get("https://osu.ppy.sh/beatmapsets/" + str(beatmap_id) + "/download", cookies={
            OsuAPIClient.__SESSION_COOKIE__: self.__session,
        })
        
        if cache:
            if not os.path.exists(OsuAPIClient.__CACHE_DIR__):
                os.makedirs(OsuAPIClient.__CACHE_DIR__)
            
            with open(cache_path, "wb") as w:
                w.write(r.content)
        
        return BeatmapSet(io.BytesIO(r.content))


if __name__ == "__main__":
    pass
    # client = OsuAPIClient("ssttevee", "37mem!9$FOO*0j25Wt9r@nn$")
    # for beatmap_result in client.search():
    
    # bms = client.download_beatmap("767253")
    
    # a = bms.audio()
    
    # a.data.tofile()
    # bms.map().encoded_timing_points().astype('float32').tofile()
